from __future__ import annotations

import re
from typing import Any


_CONDA_BUILD_PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)=(?P<version>[^=\s]+)=(?P<build>[^=\s]+)$"
)
_NATIVE_PACKAGE_NAMES = {
    "ld_impl_linux-64": "ld_impl_linux-aarch64",
}


def _relax_conda_match_spec(spec: str) -> str:
    match = _CONDA_BUILD_PIN.fullmatch(spec)
    if match:
        name = _NATIVE_PACKAGE_NAMES.get(match.group("name"), match.group("name"))
        return f"{name}={match.group('version')}"

    for x86_name, arm_name in _NATIVE_PACKAGE_NAMES.items():
        if spec == x86_name or spec.startswith(f"{x86_name}="):
            return arm_name + spec[len(x86_name) :]
    return spec


def relax_conda_lock_for_arm64(text: str) -> tuple[str, int]:
    """Remove architecture-specific build pins from direct Conda dependencies."""
    lines = text.splitlines(keepends=True)
    in_dependencies = False
    dependencies_indent = 0
    changes = 0

    for index, line in enumerate(lines):
        content = line.rstrip("\r\n")
        stripped = content.strip()
        indent = len(content) - len(content.lstrip())

        if stripped == "dependencies:":
            in_dependencies = True
            dependencies_indent = indent
            continue

        if not in_dependencies:
            continue
        if stripped and indent <= dependencies_indent:
            in_dependencies = False
            continue
        if indent != dependencies_indent + 2 or not stripped.startswith("- "):
            continue

        spec = stripped[2:].strip()
        if not spec or spec.endswith(":"):
            continue

        relaxed = _relax_conda_match_spec(spec)
        if relaxed == spec:
            continue

        newline = line[len(content) :]
        lines[index] = f"{' ' * indent}- {relaxed}{newline}"
        changes += 1

    return "".join(lines), changes


def adapt_test_spec_for_arm64(test_spec: Any) -> int:
    if getattr(test_spec, "arch", None) != "arm64":
        return 0

    adapted_commands = []
    total_changes = 0
    for command in test_spec.env_script_list:
        adapted, changes = relax_conda_lock_for_arm64(command)
        adapted_commands.append(adapted)
        total_changes += changes

    if total_changes:
        test_spec.env_script_list = adapted_commands
    return total_changes
