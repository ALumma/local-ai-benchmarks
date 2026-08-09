from __future__ import annotations

import shlex
from typing import Any


_STALE_BRANCH_HINTS = {
    ("sympy/sympy", "1.7"): "1.7",
}


def remove_stale_repo_branch_hint(test_spec: Any) -> int:
    stale_branch = _STALE_BRANCH_HINTS.get((test_spec.repo, test_spec.version))
    if stale_branch is None:
        return 0

    adapted_commands = []
    changes = 0
    for command in test_spec.repo_script_list:
        tokens = shlex.split(command)
        if tokens[:2] != ["git", "clone"]:
            adapted_commands.append(command)
            continue

        try:
            branch_index = tokens.index("--branch")
        except ValueError:
            adapted_commands.append(command)
            continue

        if branch_index + 1 >= len(tokens) or tokens[branch_index + 1] != stale_branch:
            adapted_commands.append(command)
            continue

        del tokens[branch_index : branch_index + 2]
        if "--single-branch" in tokens:
            tokens.remove("--single-branch")
        adapted_commands.append(shlex.join(tokens))
        changes += 1

    if changes:
        test_spec.repo_script_list = adapted_commands
    return changes
