"""Versioned benchmark profile locks shipped as package resources."""

from __future__ import annotations

from importlib.resources import files
from typing import Annotated

import yaml
from pydantic import Field, field_validator, model_validator

from kairyu.evaluation.schemas import (
    BenchmarkProfile,
    FrozenModel,
    ProtocolSignature,
)


class ProfileLock(FrozenModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    description: str
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    dataset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    split: str = Field(min_length=1)
    expected_full_count: Annotated[int, Field(ge=0)]
    data_license: str = Field(min_length=1)
    gated: bool
    harness_name: str = Field(min_length=1)
    harness_repository: str = Field(pattern=r"^https://\S+$")
    harness_version: str = Field(min_length=1)
    harness_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evalscope_wheel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_layer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    repeats: Annotated[int, Field(ge=1)]
    few_shot: Annotated[int, Field(ge=0)]
    shuffle_algorithm: str = Field(min_length=1)
    preprocess_version: str = Field(min_length=1)
    answer_parser_version: str = Field(min_length=1)
    metric_implementation: str = Field(min_length=1)
    source_urls: tuple[str, ...]
    unresolved_fields: tuple[str, ...] = ()

    @field_validator("source_urls")
    @classmethod
    def _source_urls_are_https(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not url.startswith("https://") for url in value):
            raise ValueError("profile source URLs must be non-empty HTTPS URLs")
        return value

    def to_profile(self, benchmark_id: str) -> BenchmarkProfile:
        """Convert the resource lock to the canonical protocol schema."""

        adapter_configuration = {
            "answer_parser": self.answer_parser_version,
            "choice_labels": ["A", "B", "C", "D"],
            "compatibility_layer_sha256": self.compatibility_layer_sha256,
            "few_shot": self.few_shot,
            "upstream_repository": self.harness_repository,
            "preprocess": self.preprocess_version,
            "prompt_sha256": self.prompt_sha256,
            "repeats": self.repeats,
            "seed": self.seed,
            "shuffle": self.shuffle_algorithm,
        }
        protocol = ProtocolSignature(
            benchmark_id=benchmark_id,
            benchmark_version="gpqa-diamond-evalscope-v1.8.1",
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            split=self.split,
            sample_filter={"mode": "full"},
            harness_name=self.harness_name,
            harness_version=self.harness_version,
            harness_commit=self.harness_commit,
            dependency_lock_sha256=self.dependency_lock_sha256,
            prompt_version=self.prompt_version,
            modalities=("text",),
            tools=(),
            web_access=False,
            retries=0,
            generation_parameters={
                "temperature": 0.0,
                "repeats": self.repeats,
                "seed": self.seed,
            },
            metric_implementation=self.metric_implementation,
            adapter_configuration=adapter_configuration,
            dependency_compatibility_patches=(
                f"kairyu.evaluation.adapters.gpqa_v181@sha256:{self.compatibility_layer_sha256}",
            ),
            unresolved_fields=self.unresolved_fields,
        )
        return BenchmarkProfile(
            name=self.name,
            benchmark_id=benchmark_id,
            description=self.description,
            protocol=protocol,
            expected_full_count=self.expected_full_count,
            source_urls=self.source_urls,
        )


class BenchmarkProfileResource(FrozenModel):
    schema_version: Annotated[int, Field(ge=1)]
    benchmark_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    profiles: tuple[ProfileLock, ...]

    @model_validator(mode="after")
    def _profile_names_are_unique(self) -> BenchmarkProfileResource:
        names = [profile.name for profile in self.profiles]
        if len(set(names)) != len(names):
            raise ValueError("benchmark profile names must be unique")
        return self


def load_profile_resource(benchmark_id: str) -> BenchmarkProfileResource:
    """Load and validate one checked-in profile resource without network access."""

    resource = files("kairyu.evaluation").joinpath("resources", "profiles", f"{benchmark_id}.yaml")
    if not resource.is_file():
        raise KeyError(f"no profile resource for {benchmark_id!r}")
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    loaded = BenchmarkProfileResource.model_validate(payload)
    if loaded.benchmark_id != benchmark_id:
        raise ValueError("profile resource benchmark ID does not match its filename")
    return loaded


def load_profiles(benchmark_id: str) -> tuple[BenchmarkProfile, ...]:
    resource = load_profile_resource(benchmark_id)
    return tuple(profile.to_profile(benchmark_id) for profile in resource.profiles)


def get_profile_lock(benchmark_id: str, profile_name: str) -> ProfileLock:
    resource = load_profile_resource(benchmark_id)
    for profile in resource.profiles:
        if profile.name == profile_name:
            return profile
    available = ", ".join(profile.name for profile in resource.profiles)
    raise KeyError(f"unknown profile {profile_name!r} for {benchmark_id!r}; available: {available}")


__all__ = [
    "BenchmarkProfileResource",
    "ProfileLock",
    "get_profile_lock",
    "load_profile_resource",
    "load_profiles",
]
