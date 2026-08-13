"""Narrow Harbor terminus-2 command-boundary compatibility adapter."""

from __future__ import annotations

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2

_STANDALONE_CONTROL_KEYSTROKES = frozenset({"C-c", "C-d"})


def terminated_keystrokes(keystrokes: str) -> str:
    """Honor terminus-2's documented newline contract for shell commands."""
    if keystrokes.endswith("\n") or keystrokes in _STANDALONE_CONTROL_KEYSTROKES:
        return keystrokes
    return keystrokes + "\n"


class KairyuTerminus2(Terminus2):
    """Official terminus-2 with command execution boundaries made atomic."""

    async def _execute_commands(self, commands, session):
        normalized = [
            Command(
                keystrokes=terminated_keystrokes(command.keystrokes),
                duration_sec=command.duration_sec,
            )
            for command in commands
        ]
        return await super()._execute_commands(normalized, session)
