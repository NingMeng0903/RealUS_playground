from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class NamedRegistry(Generic[T]):
    entries: dict[str, T] = field(default_factory=dict)

    def register(self, name: str, value: T, *, overwrite: bool = False) -> T:
        if not overwrite and name in self.entries:
            raise KeyError(f"Registry entry already exists: {name}")
        self.entries[name] = value
        return value

    def get(self, name: str) -> T:
        if name not in self.entries:
            raise KeyError(f"Unknown registry entry: {name}")
        return self.entries[name]

    def list_names(self) -> list[str]:
        return sorted(self.entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.entries


@dataclass
class PlatformRegistry:
    embodiments: NamedRegistry[object] = field(default_factory=NamedRegistry)
    scenes: NamedRegistry[object] = field(default_factory=NamedRegistry)
    observation_specs: NamedRegistry[object] = field(default_factory=NamedRegistry)
    action_specs: NamedRegistry[object] = field(default_factory=NamedRegistry)
    policy_adapters: NamedRegistry[object] = field(default_factory=NamedRegistry)
    runtime_backends: NamedRegistry[object] = field(default_factory=NamedRegistry)
    reference_providers: NamedRegistry[object] = field(default_factory=NamedRegistry)
    reward_adapters: NamedRegistry[object] = field(default_factory=NamedRegistry)
    task_adapters: NamedRegistry[object] = field(default_factory=NamedRegistry)
    runtime_batch_adapters: NamedRegistry[object] = field(default_factory=NamedRegistry)

    def summary(self) -> dict[str, list[str]]:
        return {
            "embodiments": self.embodiments.list_names(),
            "scenes": self.scenes.list_names(),
            "observation_specs": self.observation_specs.list_names(),
            "action_specs": self.action_specs.list_names(),
            "policy_adapters": self.policy_adapters.list_names(),
            "runtime_backends": self.runtime_backends.list_names(),
            "reference_providers": self.reference_providers.list_names(),
            "reward_adapters": self.reward_adapters.list_names(),
            "task_adapters": self.task_adapters.list_names(),
            "runtime_batch_adapters": self.runtime_batch_adapters.list_names(),
        }
