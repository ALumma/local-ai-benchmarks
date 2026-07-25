#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import logging
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from lm_eval._cli import HarnessCLI
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import setup_logging


eval_logger = logging.getLogger(__name__)


def seconds_from_ns(value: int | float | None) -> float | None:
    if value in (None, 0):
        return None
    return float(value) / 1_000_000_000


def rate(count: int | None, seconds: float | None) -> float | None:
    if not count or not seconds or seconds <= 0:
        return None
    return count / seconds


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@register_model("ollama-chat-timed")
class TimedOllamaChatCompletion(LocalChatCompletion):
    """Ollama native streaming chat adapter with per-request timing logs.

    This returns an OpenAI-like response object to lm-eval so the normal scoring,
    sample logging, and result aggregation path remains unchanged.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/api/chat",
        perf_log_path: str | None = None,
        perf_summary_path: str | None = None,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)
        self.perf_log_path = Path(perf_log_path) if perf_log_path else None
        self.perf_summary_path = Path(perf_summary_path) if perf_summary_path else None
        self._perf_rows: list[dict[str, Any]] = []
        self._perf_lock = threading.Lock()
        self._request_index = 0
        if self.perf_log_path:
            self.perf_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.perf_summary_path:
            self.perf_summary_path.parent.mkdir(parents=True, exist_ok=True)
        if self._concurrent > 1:
            eval_logger.warning(
                "Streaming timing requires serial requests. Defaulting num_concurrent to 1."
            )
            self._concurrent = 1

    def _create_payload(
        self,
        messages,
        generate=False,
        gen_kwargs: dict | None = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert generate, "TimedOllamaChatCompletion only supports generation."
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), "TimedOllamaChatCompletion expects messages as list[dict]."

        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if stop is None:
            stop = []
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
            "stop": list(stop)[:4],
        }
        options.update(gen_kwargs)
        return {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }

    def model_call(
        self,
        messages,
        *,
        generate: bool = True,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> dict:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        message_payload = self.create_message(messages)
        payload = self._create_payload(
            message_payload,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )

        with self._perf_lock:
            self._request_index += 1
            request_index = self._request_index

        started_at = datetime.now().isoformat(timespec="seconds")
        start = time.monotonic()
        first_chunk_at = None
        first_content_at = None
        content_parts = []
        final = {}

        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.header,
            verify=self.verify_certificate,
            stream=True,
            timeout=self.timeout,
        )
        if not response.ok:
            eval_logger.warning(
                "Ollama request failed with error message: %s. Retrying...",
                response.text,
            )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            now = time.monotonic()
            if not line:
                continue
            if first_chunk_at is None:
                first_chunk_at = now
            chunk = json.loads(line)
            content = chunk.get("message", {}).get("content") or ""
            if content:
                if first_content_at is None:
                    first_content_at = now
                content_parts.append(content)
            if chunk.get("done"):
                final = chunk
                break

        end = time.monotonic()
        text = "".join(content_parts)
        row = self._perf_row(
            request_index=request_index,
            started_at=started_at,
            wall_seconds=end - start,
            first_chunk_seconds=first_chunk_at - start if first_chunk_at else None,
            ttft_seconds=first_content_at - start if first_content_at else None,
            payload=payload,
            response_text=text,
            final=final,
        )
        self._record_perf(row)

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                }
            ]
        }

    def _perf_row(
        self,
        *,
        request_index: int,
        started_at: str,
        wall_seconds: float,
        first_chunk_seconds: float | None,
        ttft_seconds: float | None,
        payload: dict,
        response_text: str,
        final: dict,
    ) -> dict[str, Any]:
        eval_count = final.get("eval_count")
        eval_duration_seconds = seconds_from_ns(final.get("eval_duration"))
        prompt_eval_count = final.get("prompt_eval_count")
        prompt_eval_duration_seconds = seconds_from_ns(
            final.get("prompt_eval_duration")
        )
        load_duration_seconds = seconds_from_ns(final.get("load_duration"))
        total_duration_seconds = seconds_from_ns(final.get("total_duration"))
        continuing_seconds = (
            wall_seconds - ttft_seconds if ttft_seconds is not None else None
        )
        continuing_count = (
            max(eval_count - 1, 0) if isinstance(eval_count, int) else None
        )
        messages = payload["messages"]
        return {
            "request_index": request_index,
            "started_at": started_at,
            "model": payload["model"],
            "messages_sha256": hashlib.sha256(
                stable_json(messages).encode("utf-8")
            ).hexdigest(),
            "user_prompt_preview": next(
                (
                    m.get("content", "")[:240]
                    for m in reversed(messages)
                    if m.get("role") == "user"
                ),
                "",
            ),
            "prompt_chars": sum(len(m.get("content", "")) for m in messages),
            "wall_seconds": wall_seconds,
            "time_to_first_chunk_seconds": first_chunk_seconds,
            "time_to_first_token_seconds": ttft_seconds,
            "client_continuing_tokens_per_second": rate(
                continuing_count, continuing_seconds
            ),
            "ollama_eval_tokens_per_second": rate(eval_count, eval_duration_seconds),
            "eval_count": eval_count,
            "eval_duration_seconds": eval_duration_seconds,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_seconds": prompt_eval_duration_seconds,
            "prompt_eval_tokens_per_second": rate(
                prompt_eval_count, prompt_eval_duration_seconds
            ),
            "load_duration_seconds": load_duration_seconds,
            "total_duration_seconds": total_duration_seconds,
            "response_chars": len(response_text),
            "response_sha256": hashlib.sha256(
                response_text.encode("utf-8")
            ).hexdigest(),
            "response_preview": response_text[-240:],
        }

    def _record_perf(self, row: dict[str, Any]) -> None:
        with self._perf_lock:
            self._perf_rows.append(row)
            if self.perf_log_path:
                with self.perf_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if self.perf_summary_path:
                summary = self._summary()
                tmp = self.perf_summary_path.with_suffix(
                    self.perf_summary_path.suffix + ".tmp"
                )
                tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                tmp.replace(self.perf_summary_path)

    def _summary(self) -> dict[str, Any]:
        rows = self._perf_rows
        return {
            "model": self.model,
            "base_url": self.base_url,
            "sample_count": len(rows),
            "avg_time_to_first_token_seconds": mean(
                [r["time_to_first_token_seconds"] for r in rows]
            ),
            "avg_client_continuing_tokens_per_second": mean(
                [r["client_continuing_tokens_per_second"] for r in rows]
            ),
            "avg_ollama_eval_tokens_per_second": mean(
                [r["ollama_eval_tokens_per_second"] for r in rows]
            ),
            "avg_prompt_eval_tokens_per_second": mean(
                [r["prompt_eval_tokens_per_second"] for r in rows]
            ),
            "avg_load_duration_seconds": mean([r["load_duration_seconds"] for r in rows]),
            "avg_wall_seconds": mean([r["wall_seconds"] for r in rows]),
        }


@register_model("openai-chat-timed")
class TimedOpenAIChatCompletion(LocalChatCompletion):
    """OpenAI-compatible streaming chat adapter with per-request timing logs.

    Use this for vLLM/SGLang endpoints. Accuracy still flows through lm-eval's
    normal scoring path. Performance logs use client-side streaming timestamps
    plus OpenAI-compatible usage fields when the server returns them.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1/chat/completions",
        perf_log_path: str | None = None,
        perf_summary_path: str | None = None,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)
        self.perf_log_path = Path(perf_log_path) if perf_log_path else None
        self.perf_summary_path = Path(perf_summary_path) if perf_summary_path else None
        self._perf_rows: list[dict[str, Any]] = []
        self._perf_lock = threading.Lock()
        self._request_index = 0
        if self.perf_log_path:
            self.perf_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.perf_summary_path:
            self.perf_summary_path.parent.mkdir(parents=True, exist_ok=True)
        if self._concurrent > 1:
            eval_logger.warning(
                "Streaming timing requires serial requests. Defaulting num_concurrent to 1."
            )
            self._concurrent = 1

    def _create_payload(
        self,
        messages,
        generate=False,
        gen_kwargs: dict | None = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert generate, "TimedOpenAIChatCompletion only supports generation."
        assert isinstance(messages, list) and all(
            isinstance(m, dict) for m in messages
        ), "TimedOpenAIChatCompletion expects messages as list[dict]."

        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if stop is None:
            stop = []
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": list(stop)[:4],
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            **gen_kwargs,
        }

    def model_call(
        self,
        messages,
        *,
        generate: bool = True,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> dict:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        message_payload = self.create_message(messages)
        payload = self._create_payload(
            message_payload,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )

        with self._perf_lock:
            self._request_index += 1
            request_index = self._request_index

        started_at = datetime.now().isoformat(timespec="seconds")
        start = time.monotonic()
        first_chunk_at = None
        first_content_at = None
        content_parts = []
        usage = {}
        chunk_count = 0

        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.header,
            verify=self.verify_certificate,
            stream=True,
            timeout=self.timeout,
        )
        if not response.ok:
            eval_logger.warning(
                "OpenAI-compatible request failed with error message: %s. Retrying...",
                response.text,
            )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            now = time.monotonic()
            if not line:
                continue
            line = line.strip()
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break
            if first_chunk_at is None:
                first_chunk_at = now
            chunk_count += 1
            chunk = json.loads(line)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if first_content_at is None:
                        first_content_at = now
                    content_parts.append(content)

        end = time.monotonic()
        text = "".join(content_parts)
        row = self._perf_row(
            request_index=request_index,
            started_at=started_at,
            wall_seconds=end - start,
            first_chunk_seconds=first_chunk_at - start if first_chunk_at else None,
            ttft_seconds=first_content_at - start if first_content_at else None,
            payload=payload,
            response_text=text,
            usage=usage,
            chunk_count=chunk_count,
        )
        self._record_perf(row)

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                }
            ]
        }

    def _perf_row(
        self,
        *,
        request_index: int,
        started_at: str,
        wall_seconds: float,
        first_chunk_seconds: float | None,
        ttft_seconds: float | None,
        payload: dict,
        response_text: str,
        usage: dict[str, Any],
        chunk_count: int,
    ) -> dict[str, Any]:
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        continuing_seconds = (
            wall_seconds - ttft_seconds if ttft_seconds is not None else None
        )
        continuing_count = (
            max(completion_tokens - 1, 0)
            if isinstance(completion_tokens, int)
            else None
        )
        messages = payload["messages"]
        return {
            "request_index": request_index,
            "started_at": started_at,
            "model": payload["model"],
            "messages_sha256": hashlib.sha256(
                stable_json(messages).encode("utf-8")
            ).hexdigest(),
            "user_prompt_preview": next(
                (
                    m.get("content", "")[:240]
                    for m in reversed(messages)
                    if m.get("role") == "user"
                ),
                "",
            ),
            "prompt_chars": sum(len(m.get("content", "")) for m in messages),
            "wall_seconds": wall_seconds,
            "time_to_first_chunk_seconds": first_chunk_seconds,
            "time_to_first_token_seconds": ttft_seconds,
            "client_continuing_tokens_per_second": rate(
                continuing_count, continuing_seconds
            ),
            "openai_completion_tokens_per_second": rate(
                completion_tokens, wall_seconds
            ),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": usage.get("total_tokens"),
            "stream_chunk_count": chunk_count,
            "stream_chunks_per_second": rate(chunk_count, wall_seconds),
            "response_chars": len(response_text),
            "response_sha256": hashlib.sha256(
                response_text.encode("utf-8")
            ).hexdigest(),
            "response_preview": response_text[-240:],
        }

    def _record_perf(self, row: dict[str, Any]) -> None:
        with self._perf_lock:
            self._perf_rows.append(row)
            if self.perf_log_path:
                with self.perf_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if self.perf_summary_path:
                summary = self._summary()
                tmp = self.perf_summary_path.with_suffix(
                    self.perf_summary_path.suffix + ".tmp"
                )
                tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                tmp.replace(self.perf_summary_path)

    def _summary(self) -> dict[str, Any]:
        rows = self._perf_rows
        return {
            "model": self.model,
            "base_url": self.base_url,
            "sample_count": len(rows),
            "avg_time_to_first_token_seconds": mean(
                [r["time_to_first_token_seconds"] for r in rows]
            ),
            "avg_client_continuing_tokens_per_second": mean(
                [r["client_continuing_tokens_per_second"] for r in rows]
            ),
            "avg_openai_completion_tokens_per_second": mean(
                [r["openai_completion_tokens_per_second"] for r in rows]
            ),
            "avg_stream_chunks_per_second": mean(
                [r["stream_chunks_per_second"] for r in rows]
            ),
            "avg_wall_seconds": mean([r["wall_seconds"] for r in rows]),
        }


def main() -> None:
    setup_logging()
    parser = HarnessCLI()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    main()
