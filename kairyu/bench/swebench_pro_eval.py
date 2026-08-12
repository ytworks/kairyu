"""Run the pinned SWE-bench Pro evaluator with bounded Docker image storage."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType


def _load_evaluator(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kairyu_swebench_pro_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load SWE-bench Pro evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ephemeral_eval(
    module: ModuleType,
    original: Callable,
) -> Callable:
    def wrapped(
        patch,
        sample,
        output_dir,
        dockerhub_username,
        scripts_dir,
        prefix="",
        redo=False,
        block_network=False,
        docker_platform=None,
    ):
        image = module.get_dockerhub_image_uri(
            sample["instance_id"], dockerhub_username, sample.get("repo", "")
        )
        client = module.docker.from_env()
        remove_image = False
        try:
            client.images.get(image)
        except Exception:
            remove_image = True
        finally:
            client.close()
        try:
            return original(
                patch,
                sample,
                output_dir,
                dockerhub_username,
                scripts_dir,
                prefix=prefix,
                redo=redo,
                block_network=block_network,
                docker_platform=docker_platform,
            )
        finally:
            if remove_image:
                client = module.docker.from_env()
                try:
                    for attempt in range(10):
                        try:
                            client.images.remove(image)
                            break
                        except Exception as error:
                            if attempt == 9:
                                print(
                                    "Warning: could not remove ephemeral task image "
                                    f"{image}: {error}",
                                    file=sys.stderr,
                                )
                            else:
                                time.sleep(1)
                finally:
                    client.close()

    return wrapped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    args, evaluator_args = parser.parse_known_args(argv)
    evaluator = _load_evaluator(args.evaluator_script.resolve())
    evaluator.eval_with_docker = _ephemeral_eval(
        evaluator, evaluator.eval_with_docker
    )
    sys.argv = [str(args.evaluator_script), *evaluator_args]
    evaluator.main()


if __name__ == "__main__":
    main()
