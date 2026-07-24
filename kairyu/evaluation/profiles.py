"""Versioned benchmark profile locks shipped as package resources."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Annotated, Any, cast

import yaml
from pydantic import (
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError

from kairyu.evaluation.schemas import (
    BenchmarkProfile,
    FrozenModel,
    ProtocolSignature,
    freeze_json_value,
    thaw_json_value,
)

_BENCHMARK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ProfileDataError(ValueError):
    """A packaged benchmark profile resource is malformed."""


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
    evalscope_wheel_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
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
    benchmark_version: str = Field(
        default="gpqa-diamond-evalscope-v1.8.1",
        min_length=1,
    )
    modalities: tuple[str, ...] = ("text",)
    tools: tuple[str, ...] = ()
    web_access: bool = False
    retries: Annotated[int, Field(ge=0)] = 0
    judge_model: str | None = Field(default=None, min_length=1)
    judge_prompt_version: str | None = Field(default=None, min_length=1)
    generation_parameters: dict[str, JsonValue] | None = None
    adapter_configuration: dict[str, JsonValue] | None = None
    compatibility_patch_module: str | None = Field(
        default="kairyu.evaluation.adapters.gpqa_v181",
        min_length=1,
    )

    @field_validator("source_urls")
    @classmethod
    def _source_urls_are_https(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not url.startswith("https://") for url in value):
            raise ValueError("profile source URLs must be non-empty HTTPS URLs")
        return value

    @field_validator("modalities")
    @classmethod
    def _modalities_are_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("profile modalities must be non-empty")
        return value

    @field_validator("modalities", "tools")
    @classmethod
    def _protocol_lists_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("protocol list values must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("protocol list values must be unique")
        return value

    @field_validator(
        "generation_parameters",
        "adapter_configuration",
    )
    @classmethod
    def _freeze_json_maps(
        cls,
        value: dict[str, JsonValue] | None,
    ) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        return cast(dict[str, JsonValue], freeze_json_value(value))

    @field_serializer("generation_parameters", "adapter_configuration")
    def _serialize_json_maps(
        self,
        value: Mapping[str, JsonValue] | None,
    ) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        return cast(dict[str, JsonValue], thaw_json_value(value))

    @model_validator(mode="after")
    def _judge_fields_are_coherent(self) -> ProfileLock:
        if (self.judge_model is None) != (self.judge_prompt_version is None):
            raise ValueError("judge model and prompt version must be set together")
        return self

    def to_profile(self, benchmark_id: str) -> BenchmarkProfile:
        """Convert the resource lock to the canonical protocol schema."""

        if benchmark_id == "gpqa-diamond" and self.evalscope_wheel_sha256 is None:
            raise ProfileDataError("GPQA profile locks must retain the pinned EvalScope wheel hash")

        pinned_adapter_configuration: dict[str, JsonValue] = {
            "compatibility_layer_sha256": self.compatibility_layer_sha256,
            "prompt_sha256": self.prompt_sha256,
            "upstream_repository": self.harness_repository,
        }
        legacy_adapter_configuration: dict[str, JsonValue] = {
            "answer_parser": self.answer_parser_version,
            "choice_labels": ["A", "B", "C", "D"],
            "few_shot": self.few_shot,
            "preprocess": self.preprocess_version,
            "repeats": self.repeats,
            "seed": self.seed,
            "shuffle": self.shuffle_algorithm,
        }
        explicit_adapter_configuration = (
            {}
            if self.adapter_configuration is None
            else cast(
                dict[str, JsonValue],
                thaw_json_value(self.adapter_configuration),
            )
        )
        for field, expected in pinned_adapter_configuration.items():
            if (
                field in explicit_adapter_configuration
                and explicit_adapter_configuration[field] != expected
            ):
                raise ProfileDataError(f"adapter configuration cannot override pinned {field!r}")
        adapter_configuration = (
            legacy_adapter_configuration
            if self.adapter_configuration is None
            else explicit_adapter_configuration
        )
        adapter_configuration.update(pinned_adapter_configuration)

        generation_parameters = (
            {
                "temperature": 0.0,
                "repeats": self.repeats,
                "seed": self.seed,
            }
            if self.generation_parameters is None
            else cast(
                dict[str, JsonValue],
                thaw_json_value(self.generation_parameters),
            )
        )
        compatibility_patches = (
            ()
            if self.compatibility_patch_module is None
            else (f"{self.compatibility_patch_module}@sha256:{self.compatibility_layer_sha256}",)
        )
        protocol = ProtocolSignature(
            benchmark_id=benchmark_id,
            benchmark_version=self.benchmark_version,
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            split=self.split,
            sample_filter={"mode": "full"},
            harness_name=self.harness_name,
            harness_version=self.harness_version,
            harness_commit=self.harness_commit,
            dependency_lock_sha256=self.dependency_lock_sha256,
            prompt_version=self.prompt_version,
            modalities=self.modalities,
            tools=self.tools,
            web_access=self.web_access,
            retries=self.retries,
            judge_model=self.judge_model,
            judge_prompt_version=self.judge_prompt_version,
            generation_parameters=generation_parameters,
            metric_implementation=self.metric_implementation,
            adapter_configuration=adapter_configuration,
            dependency_compatibility_patches=compatibility_patches,
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
    schema_version: Annotated[int, Field(ge=1, le=1)]
    benchmark_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    profiles: Annotated[tuple[ProfileLock, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _profile_names_are_unique(self) -> BenchmarkProfileResource:
        names = [profile.name for profile in self.profiles]
        if len(set(names)) != len(names):
            raise ValueError("benchmark profile names must be unique")
        if self.benchmark_id == "gpqa-diamond" and any(
            profile.evalscope_wheel_sha256 is None for profile in self.profiles
        ):
            raise ValueError("GPQA profile locks must retain the pinned EvalScope wheel hash")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_profile_resource(benchmark_id: str) -> BenchmarkProfileResource:
    """Load and validate one checked-in profile resource without network access."""

    if _BENCHMARK_ID_PATTERN.fullmatch(benchmark_id) is None:
        raise KeyError(f"invalid benchmark ID {benchmark_id!r}")
    resource = files("kairyu.evaluation").joinpath(
        "resources",
        "profiles",
        f"{benchmark_id}.yaml",
    )
    if not resource.is_file():
        raise KeyError(f"no profile resource for {benchmark_id!r}")
    try:
        text = resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProfileDataError("profile resource could not be read") from exc
    try:
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ProfileDataError("profile resource is not valid strict YAML") from exc
    if not isinstance(payload, Mapping):
        raise ProfileDataError("profile resource root must be a mapping")
    try:
        loaded = BenchmarkProfileResource.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise ProfileDataError("profile resource failed schema validation") from exc
    if loaded.benchmark_id != benchmark_id:
        raise ProfileDataError("profile resource benchmark ID does not match its filename")
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
    "ProfileDataError",
    "ProfileLock",
    "get_profile_lock",
    "load_profile_resource",
    "load_profiles",
]
