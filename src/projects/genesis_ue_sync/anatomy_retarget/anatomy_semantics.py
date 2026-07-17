"""Strict, deterministic semantics for meshes in the source anatomy asset.

The manifest intentionally supports exact mesh names plus exact collection and
tissue defaults.  It does not support regexes, token matching, or implicit name
classification: a production mesh is either covered deterministically or the
export fails.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_SEMANTIC_FIELDS = (
    "tissue_type",
    "fit_policy",
    "driver_policy",
    "compound_id",
    "side",
    "source_landmarks",
    "target_landmark_recipe",
    "quality_profile",
)
_OPTIONAL_SEMANTIC_FIELDS = ("material_group", "role")
_RECORD_FIELDS = frozenset((*REQUIRED_SEMANTIC_FIELDS, *_OPTIONAL_SEMANTIC_FIELDS))
_TOP_LEVEL_FIELDS = frozenset(
    (
        "version",
        "global_defaults",
        "collection_defaults",
        "tissue_defaults",
        "meshes",
        "quality_profiles",
        "landmark_recipes",
        "fit_policies",
        "driver_policies",
        "notes",
    )
)
_FORBIDDEN_RULE_KEYS = frozenset(("regex", "pattern", "patterns", "token", "tokens", "match"))
_VALID_SIDES = frozenset(("left", "right", "midline", "bilateral", "none"))


class SemanticManifestError(ValueError):
    """Raised when a semantics manifest cannot resolve meshes unambiguously."""


@dataclass(frozen=True)
class ResolvedMeshSemantics:
    mesh_name: str
    collections: tuple[str, ...]
    tissue_type: str
    fit_policy: str
    driver_policy: str
    compound_id: str
    side: str
    source_landmarks: tuple[str, ...]
    target_landmark_recipe: str
    quality_profile: str
    material_group: str
    role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_name": self.mesh_name,
            "collections": list(self.collections),
            "tissue_type": self.tissue_type,
            "fit_policy": self.fit_policy,
            "driver_policy": self.driver_policy,
            "compound_id": self.compound_id,
            "side": self.side,
            "source_landmarks": list(self.source_landmarks),
            "target_landmark_recipe": self.target_landmark_recipe,
            "quality_profile": self.quality_profile,
            "material_group": self.material_group,
            "role": self.role,
        }


@dataclass(frozen=True)
class AnatomySemanticManifest:
    version: int
    global_defaults: Mapping[str, Any]
    collection_defaults: Mapping[str, Mapping[str, Any]]
    tissue_defaults: Mapping[str, Mapping[str, Any]]
    meshes: Mapping[str, Mapping[str, Any]]
    quality_profiles: Mapping[str, Mapping[str, Any]]
    landmark_recipes: Mapping[str, Mapping[str, Any]]
    fit_policies: tuple[str, ...]
    driver_policies: tuple[str, ...]
    source: str
    sha256: str

    def resolve(self, mesh_name: str, collections: Sequence[str]) -> ResolvedMeshSemantics:
        name = str(mesh_name)
        collection_names = tuple(sorted({str(value) for value in collections}))
        explicit = dict(self.meshes.get(name, {}))
        matching = [
            (collection, dict(self.collection_defaults[collection]))
            for collection in collection_names
            if collection in self.collection_defaults
        ]
        if not explicit and not matching:
            raise SemanticManifestError(
                f"mesh {name!r} is unresolved: no exact mesh or collection record"
            )

        collection_values: dict[str, Any] = {}
        for field in _RECORD_FIELDS:
            candidates = [
                (collection, record[field])
                for collection, record in matching
                if field in record
            ]
            distinct = {_stable_value(value) for _collection, value in candidates}
            if len(distinct) > 1 and field not in explicit:
                detail = ", ".join(
                    f"{collection}={value!r}" for collection, value in candidates
                )
                raise SemanticManifestError(
                    f"mesh {name!r} has ambiguous collection defaults for {field}: {detail}"
                )
            if candidates and len(distinct) == 1:
                collection_values[field] = candidates[0][1]

        tissue_type = explicit.get(
            "tissue_type",
            collection_values.get("tissue_type", self.global_defaults.get("tissue_type")),
        )
        if not isinstance(tissue_type, str) or not tissue_type:
            raise SemanticManifestError(
                f"mesh {name!r} cannot select a tissue default without tissue_type"
            )
        if tissue_type not in self.tissue_defaults:
            raise SemanticManifestError(
                f"mesh {name!r} references undefined tissue_type {tissue_type!r}"
            )

        resolved: dict[str, Any] = {}
        resolved.update(self.global_defaults)
        resolved.update(self.tissue_defaults[tissue_type])
        resolved.update(collection_values)
        resolved.update(explicit)
        resolved["tissue_type"] = tissue_type
        _validate_resolved_record(
            name,
            resolved,
            quality_profiles=self.quality_profiles,
            landmark_recipes=self.landmark_recipes,
            fit_policies=self.fit_policies,
            driver_policies=self.driver_policies,
        )
        material_group = str(
            resolved.get(
                "material_group",
                resolved["compound_id"]
                if resolved["compound_id"] != "none"
                else ("soft_tissue" if resolved["fit_policy"] == "soft_volume" else "skeletal"),
            )
        )
        role = str(resolved.get("role", resolved["fit_policy"]))
        return ResolvedMeshSemantics(
            mesh_name=name,
            collections=collection_names,
            tissue_type=str(resolved["tissue_type"]),
            fit_policy=str(resolved["fit_policy"]),
            driver_policy=str(resolved["driver_policy"]),
            compound_id=str(resolved["compound_id"]),
            side=str(resolved["side"]),
            source_landmarks=tuple(str(value) for value in resolved["source_landmarks"]),
            target_landmark_recipe=str(resolved["target_landmark_recipe"]),
            quality_profile=str(resolved["quality_profile"]),
            material_group=material_group,
            role=role,
        )

    def resolve_many(
        self,
        meshes: Iterable[tuple[str, Sequence[str]]],
    ) -> dict[str, ResolvedMeshSemantics]:
        resolved: dict[str, ResolvedMeshSemantics] = {}
        for name, collections in sorted(meshes, key=lambda item: str(item[0])):
            key = str(name)
            if key in resolved:
                raise SemanticManifestError(f"duplicate source mesh name {key!r}")
            resolved[key] = self.resolve(key, collections)
        return resolved


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mapping(value: Any, *, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SemanticManifestError(f"{where} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _string_sequence(value: Any, *, where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SemanticManifestError(f"{where} must be a sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise SemanticManifestError(f"{where} may not contain empty strings")
    if len(set(result)) != len(result):
        raise SemanticManifestError(f"{where} may not contain duplicates")
    return result


def _semantic_record(value: Any, *, where: str) -> dict[str, Any]:
    record = _mapping(value, where=where)
    forbidden = sorted(set(record).intersection(_FORBIDDEN_RULE_KEYS))
    if forbidden:
        raise SemanticManifestError(
            f"{where} uses forbidden implicit matching keys: {', '.join(forbidden)}"
        )
    unknown = sorted(set(record).difference(_RECORD_FIELDS))
    if unknown:
        raise SemanticManifestError(f"{where} has unknown fields: {', '.join(unknown)}")
    if "source_landmarks" in record:
        record["source_landmarks"] = list(
            _string_sequence(record["source_landmarks"], where=f"{where}.source_landmarks")
        )
    return record


def _record_map(value: Any, *, where: str, exact_names: bool = False) -> dict[str, dict[str, Any]]:
    records = _mapping(value, where=where)
    result: dict[str, dict[str, Any]] = {}
    for key, record in records.items():
        if not key:
            raise SemanticManifestError(f"{where} may not contain an empty key")
        if exact_names and any(character in key for character in ("*", "?")):
            raise SemanticManifestError(
                f"{where}.{key} must be an exact name; wildcard matching is forbidden"
            )
        result[key] = _semantic_record(record, where=f"{where}.{key}")
    return result


def _validate_resolved_record(
    mesh_name: str,
    record: Mapping[str, Any],
    *,
    quality_profiles: Mapping[str, Mapping[str, Any]],
    landmark_recipes: Mapping[str, Mapping[str, Any]],
    fit_policies: Sequence[str],
    driver_policies: Sequence[str],
) -> None:
    missing = [field for field in REQUIRED_SEMANTIC_FIELDS if field not in record]
    if missing:
        raise SemanticManifestError(
            f"mesh {mesh_name!r} is missing required semantics: {', '.join(missing)}"
        )
    for field in REQUIRED_SEMANTIC_FIELDS:
        if field == "source_landmarks":
            landmarks = _string_sequence(
                record[field], where=f"mesh {mesh_name!r}.{field}"
            )
            if not landmarks:
                raise SemanticManifestError(
                    f"mesh {mesh_name!r}.source_landmarks must not be empty"
                )
        elif not isinstance(record[field], str) or not record[field]:
            raise SemanticManifestError(
                f"mesh {mesh_name!r}.{field} must be a non-empty string"
            )
    if record["side"] not in _VALID_SIDES:
        raise SemanticManifestError(
            f"mesh {mesh_name!r}.side must be one of {sorted(_VALID_SIDES)}"
        )
    if record["quality_profile"] not in quality_profiles:
        raise SemanticManifestError(
            f"mesh {mesh_name!r} references undefined quality_profile "
            f"{record['quality_profile']!r}"
        )
    if record["target_landmark_recipe"] not in landmark_recipes:
        raise SemanticManifestError(
            f"mesh {mesh_name!r} references undefined target_landmark_recipe "
            f"{record['target_landmark_recipe']!r}"
        )
    if fit_policies and record["fit_policy"] not in fit_policies:
        raise SemanticManifestError(
            f"mesh {mesh_name!r} references undefined fit_policy {record['fit_policy']!r}"
        )
    if driver_policies and record["driver_policy"] not in driver_policies:
        raise SemanticManifestError(
            f"mesh {mesh_name!r} references undefined driver_policy "
            f"{record['driver_policy']!r}"
        )


def parse_anatomy_semantics(
    payload: Mapping[str, Any],
    *,
    source: str = "<memory>",
    source_bytes: bytes | None = None,
) -> AnatomySemanticManifest:
    root = _mapping(payload, where="semantic manifest")
    unknown = sorted(set(root).difference(_TOP_LEVEL_FIELDS))
    if unknown:
        raise SemanticManifestError(
            f"semantic manifest has unknown fields: {', '.join(unknown)}"
        )
    version = root.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SemanticManifestError("semantic manifest version must be a positive integer")

    quality_profiles_raw = _mapping(root.get("quality_profiles"), where="quality_profiles")
    if not quality_profiles_raw:
        raise SemanticManifestError("quality_profiles must define at least one profile")
    quality_profiles = {
        name: _mapping(value, where=f"quality_profiles.{name}")
        for name, value in quality_profiles_raw.items()
    }
    landmark_recipes_raw = _mapping(root.get("landmark_recipes"), where="landmark_recipes")
    if not landmark_recipes_raw:
        raise SemanticManifestError("landmark_recipes must define at least one recipe")
    landmark_recipes = {
        name: _mapping(value, where=f"landmark_recipes.{name}")
        for name, value in landmark_recipes_raw.items()
    }
    fit_policies = _string_sequence(root.get("fit_policies", ()), where="fit_policies")
    driver_policies = _string_sequence(
        root.get("driver_policies", ()), where="driver_policies"
    )
    global_defaults = _semantic_record(
        root.get("global_defaults"), where="global_defaults"
    )
    collection_defaults = _record_map(
        root.get("collection_defaults"),
        where="collection_defaults",
        exact_names=True,
    )
    tissue_defaults = _record_map(
        root.get("tissue_defaults"), where="tissue_defaults", exact_names=True
    )
    meshes = _record_map(root.get("meshes"), where="meshes", exact_names=True)
    if not collection_defaults and not meshes:
        raise SemanticManifestError(
            "semantic manifest must cover meshes by exact collection or mesh records"
        )
    for tissue, record in tissue_defaults.items():
        if "tissue_type" in record and record["tissue_type"] != tissue:
            raise SemanticManifestError(
                f"tissue_defaults.{tissue}.tissue_type must equal {tissue!r}"
            )

    if source_bytes is None:
        source_bytes = json.dumps(
            root, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    return AnatomySemanticManifest(
        version=version,
        global_defaults=global_defaults,
        collection_defaults=collection_defaults,
        tissue_defaults=tissue_defaults,
        meshes=meshes,
        quality_profiles=quality_profiles,
        landmark_recipes=landmark_recipes,
        fit_policies=fit_policies,
        driver_policies=driver_policies,
        source=str(source),
        sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def _split_inline_yaml(value: str) -> list[str]:
    items: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if quote is not None:
            if character == quote and (index == 0 or value[index - 1] != "\\"):
                quote = None
        elif character in ("'", '"'):
            quote = character
        elif character in ("[", "{"):
            depth += 1
        elif character in ("]", "}"):
            depth -= 1
        elif character == "," and depth == 0:
            items.append(value[start:index].strip())
            start = index + 1
    items.append(value[start:].strip())
    return items


def _simple_yaml_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in ("null", "~"):
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text == "[]":
        return []
    if text == "{}":
        return {}
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        return [] if not body else [
            _simple_yaml_scalar(item) for item in _split_inline_yaml(body)
        ]
    if text.startswith("{") and text.endswith("}"):
        body = text[1:-1].strip()
        result: dict[str, Any] = {}
        if body:
            for item in _split_inline_yaml(body):
                key, separator, nested = item.partition(":")
                if not separator:
                    raise SemanticManifestError(
                        f"invalid inline YAML mapping item {item!r}"
                    )
                result[str(_simple_yaml_scalar(key))] = _simple_yaml_scalar(nested)
        return result
    if text.startswith(("'", '"')):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise SemanticManifestError(f"invalid quoted YAML scalar {text!r}") from exc
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_simple_yaml(text: str) -> Mapping[str, Any]:
    """Parse the manifest subset when Blender has no PyYAML installation.

    The supported subset is deliberately small and deterministic: indentation
    mappings, scalar lists, inline lists/mappings, and literal text blocks.
    Anchors, tags, merge keys, and implicit match rules are not accepted.
    """

    physical_lines = text.splitlines()
    tokens: list[tuple[int, str, int]] = []
    for line_number, raw in enumerate(physical_lines, start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise SemanticManifestError(
                f"tabs are forbidden in semantic YAML indentation (line {line_number})"
            )
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        tokens.append((indent, raw[indent:], line_number))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(tokens):
            return {}, index
        is_list = tokens[index][1].startswith("- ")
        container: Any = [] if is_list else {}
        while index < len(tokens):
            current_indent, content, line_number = tokens[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise SemanticManifestError(
                    f"unexpected YAML indentation at line {line_number}"
                )
            if is_list:
                if not content.startswith("- "):
                    raise SemanticManifestError(
                        f"mixed YAML list and mapping at line {line_number}"
                    )
                item = content[2:].strip()
                if not item:
                    if index + 1 >= len(tokens) or tokens[index + 1][0] <= indent:
                        container.append(None)
                        index += 1
                    else:
                        nested, index = parse_block(index + 1, tokens[index + 1][0])
                        container.append(nested)
                else:
                    container.append(_simple_yaml_scalar(item))
                    index += 1
                continue
            if content.startswith("- "):
                raise SemanticManifestError(
                    f"mixed YAML mapping and list at line {line_number}"
                )
            key_text, separator, value_text = content.partition(":")
            if not separator:
                raise SemanticManifestError(
                    f"expected YAML mapping entry at line {line_number}"
                )
            key = str(_simple_yaml_scalar(key_text.strip()))
            if not key or key in container:
                raise SemanticManifestError(
                    f"empty or duplicate YAML key {key!r} at line {line_number}"
                )
            value_text = value_text.strip()
            if value_text in ("|", ">"):
                block_lines: list[str] = []
                index += 1
                while index < len(tokens) and tokens[index][0] > indent:
                    block_lines.append(tokens[index][1])
                    index += 1
                container[key] = "\n".join(block_lines)
            elif value_text:
                container[key] = _simple_yaml_scalar(value_text)
                index += 1
            elif index + 1 < len(tokens) and tokens[index + 1][0] > indent:
                nested, index = parse_block(index + 1, tokens[index + 1][0])
                container[key] = nested
            else:
                container[key] = {}
                index += 1
        return container, index

    if not tokens:
        return {}
    if tokens[0][0] != 0:
        raise SemanticManifestError("semantic YAML must start at indentation zero")
    parsed, final_index = parse_block(0, 0)
    if final_index != len(tokens) or not isinstance(parsed, Mapping):
        raise SemanticManifestError("semantic YAML root must be a mapping")
    return parsed


def load_anatomy_semantics(path: Path | str) -> AnatomySemanticManifest:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"required anatomy semantic manifest not found: {source_path}")
    raw = source_path.read_bytes()
    text = raw.decode("utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        payload = _parse_simple_yaml(text)
    else:
        try:
            payload = yaml.safe_load(text)
        except Exception as exc:
            raise SemanticManifestError(
                f"cannot parse anatomy semantic manifest {source_path}: {exc}"
            ) from exc
    return parse_anatomy_semantics(
        _mapping(payload, where=f"semantic manifest {source_path}"),
        source=str(source_path),
        source_bytes=raw,
    )
