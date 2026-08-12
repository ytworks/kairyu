"""Small mini-swe-agent compatibility shims loaded only by its subprocess."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
import uuid

from minisweagent.environments import _ENVIRONMENT_MAPPING
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

_OPENAI_CHAT_MESSAGE_FIELDS = frozenset(
    {"role", "content", "name", "tool_call_id", "tool_calls"}
)
_EPHEMERAL_DOCKER_ENVIRONMENT = (
    "kairyu.bench.mini_swe_agent.EphemeralDockerEnvironment"
)


class EphemeralDockerEnvironment(DockerEnvironment):
    """Remove task images pulled by this environment after the task finishes."""

    def __init__(self, **kwargs):
        self._container_name: str | None = None
        self._cleanup_started = False
        self._remove_image = False
        super().__init__(**kwargs)

    def _run_docker(
        self,
        *args: str,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.config.executable, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _start_container(self) -> None:
        """Start the task container while retaining enough state for cleanup."""

        image_check = self._run_docker(
            "image", "inspect", self.config.image, timeout=30
        )
        self._remove_image = image_check.returncode != 0
        self._container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        command = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            self._container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug("Starting container with command: %s", shlex.join(command))
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.pull_timeout,
                check=True,
            )
        except BaseException:
            self.cleanup()
            raise
        self.container_id = result.stdout.strip()
        self.logger.info(
            "Started container %s with ID %s",
            self._container_name,
            self.container_id,
        )

    def cleanup(self) -> None:
        """Remove the container and any task image that this run pulled."""

        if self._cleanup_started:
            return
        self._cleanup_started = True
        reference = self.container_id or self._container_name
        if reference is not None:
            stopped = self._run_docker("stop", "--time", "1", reference, timeout=15)
            if stopped.returncode != 0:
                self._run_docker("rm", "-f", reference, timeout=15)
        self.container_id = None
        if not self._remove_image:
            return
        for attempt in range(10):
            removed = self._run_docker(
                "image", "rm", self.config.image, timeout=180
            )
            if removed.returncode == 0:
                return
            if attempt < 9:
                time.sleep(1)
        logging.getLogger("minisweagent.environment").warning(
            "Could not remove ephemeral task image %s: %s",
            self.config.image,
            removed.stderr.strip()[-500:],
        )


class OpenAICompatTextbasedModel(LitellmTextbasedModel):
    """Keep LiteLLM response-only extensions out of later chat requests."""

    def __init__(self, **kwargs):
        # mini-swe-agent resolves the model before its environment for every
        # SWE-bench task. Replace only its built-in ``docker`` alias so its
        # per-instance image injection continues to work with our cleanup
        # environment.
        _ENVIRONMENT_MAPPING["docker"] = _EPHEMERAL_DOCKER_ENVIRONMENT
        super().__init__(**kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = super()._prepare_messages_for_api(messages)
        return [
            {
                key: value
                for key, value in message.items()
                if key in _OPENAI_CHAT_MESSAGE_FIELDS
            }
            for message in prepared
        ]


__all__ = ["EphemeralDockerEnvironment", "OpenAICompatTextbasedModel"]
