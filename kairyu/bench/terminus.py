"""Narrow Harbor terminus-2 command-boundary compatibility adapter."""

from __future__ import annotations

import shlex

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2

_STANDALONE_CONTROL_KEYSTROKES = frozenset({"C-c", "C-d"})
_MAX_TRANSPORT_KEYSTROKES = 1200
_COMMAND_TRANSPORT_ADDENDUM = """\
Kairyu command transport constraints:
- Return at most one command object in each response.
- Keep each keystrokes value below 1200 characters. Split longer work across
  later turns; inspect the result after each chunk.
- Do not use shell heredocs or commands that require embedded newlines; this
  JSON command transport can flatten those newlines before execution.
- Do not calculate or emit base64 data. To write a multiline file, use one
  single-line `printf '%s\\n' 'line 1' 'line 2' > /path/file` command for the
  first few lines, then similarly append a few lines per later turn with `>>`.
  Shell-quote every line and keep the whole command below 1200 characters.
"""


def command_transport_prompt(upstream: str) -> str:
    """Append the generic transport contract without replacing Harbor's prompt."""
    return upstream.rstrip() + "\n\n" + _COMMAND_TRANSPORT_ADDENDUM


def terminated_keystrokes(keystrokes: str) -> str:
    """Honor terminus-2's documented newline contract for shell commands."""
    if keystrokes.endswith("\n") or keystrokes in _STANDALONE_CONTROL_KEYSTROKES:
        return keystrokes
    return keystrokes + "\n"


def command_transport_error(keystrokes: str) -> str | None:
    """Reject command shapes that cannot cross this JSON transport intact."""
    if keystrokes in _STANDALONE_CONTROL_KEYSTROKES:
        return None
    if len(keystrokes) > _MAX_TRANSPORT_KEYSTROKES:
        return (
            f"keystrokes has {len(keystrokes)} characters; the transport maximum "
            f"is {_MAX_TRANSPORT_KEYSTROKES}. Split the work across later turns"
        )
    body = keystrokes[:-1] if keystrokes.endswith("\n") else keystrokes
    if "\n" in body or "\r" in body:
        return "keystrokes contains embedded newlines; use one single-line command"
    try:
        lexer = shlex.shlex(body, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = tuple(lexer)
    except ValueError:
        return "keystrokes has unbalanced shell quoting"
    if "<<" in tokens:
        return "shell heredocs are unsupported; append a few quoted lines with printf"
    return None


class KairyuTerminus2(Terminus2):
    """Official terminus-2 with command execution boundaries made atomic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prompt_template = command_transport_prompt(self._prompt_template)

    async def _execute_commands(self, commands, session):
        for command in commands:
            error = command_transport_error(command.keystrokes)
            if error is not None:
                terminal_state = self._limit_output_length(
                    await session.get_incremental_output()
                )
                return (
                    False,
                    "Kairyu command transport rejected the command without "
                    f"executing it: {error}.\nCurrent terminal state is unchanged:\n"
                    f"{terminal_state}",
                )
        normalized = [
            Command(
                keystrokes=terminated_keystrokes(command.keystrokes),
                duration_sec=command.duration_sec,
            )
            for command in commands
        ]
        return await super()._execute_commands(normalized, session)
