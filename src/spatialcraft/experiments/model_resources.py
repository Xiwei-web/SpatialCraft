"""Content identities for every distinct local executor/scorer/knowledge model.

Preflight hashes auxiliary files only. Authorized execute hashes weight files
once per resolved path, across all roles. Runtime verifies the bound inventory,
auxiliary contents and weight stat identities before opening a journal. A new
execute/resume performs a fresh complete weight hash; no global hash cache is used.
"""

from __future__ import annotations

from pathlib import Path

from spatialcraft.models.registry import ModelConfig, load_yaml
from spatialcraft.storage.atomic_io import read_json, sha256_file

from .journal import digest

_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".gguf",
}
_IGNORED = {".cache", ".git", "__pycache__"}


def resolve_role_models(project, settings):
    aliases = {
        "executor": settings.backbone,
        "scorer": settings.roles.get("scorer", settings.backbone),
        "knowledge_builder": settings.roles.get("knowledge_builder", settings.backbone),
    }
    configs = {
        alias: ModelConfig.from_dict(
            load_yaml(Path(project) / f"configs/models/{alias}.yaml")
        )
        for alias in set(aliases.values())
    }
    return {role: configs[alias] for role, alias in aliases.items()}


def _recipe(model):
    local = model.local
    if local is None:
        return None
    return {
        "model_path": str(Path(local.path).expanduser().resolve()),
        # AutoProcessor and the metric tokenizer both use this effective path.
        "tokenizer_processor_path": str(
            Path(local.tokenizer_path or local.path).expanduser().resolve()
        ),
        "model_revision": local.extra_load_kwargs.get("revision", local.revision),
        "processor_revision": local.revision,
    }


def _inventory(recipe):
    files = {}
    model_root = Path(recipe["model_path"])
    for root in {model_root, Path(recipe["tokenizer_processor_path"])}:
        if not root.is_dir():
            raise ValueError(f"Local role model resource directory missing: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and not _IGNORED.intersection(
                path.relative_to(root).parts
            ):
                files[str(path.absolute())] = path.resolve()
    if not any(
        path.suffix in _WEIGHT_SUFFIXES
        for path in model_root.rglob("*")
        if path.is_file()
    ):
        raise ValueError(f"Local role model has no weight files: {model_root}")
    for index in model_root.rglob("*.index.json"):
        if _IGNORED.intersection(index.relative_to(model_root).parts):
            continue
        index_value = read_json(index)
        if not isinstance(index_value, dict) or "weight_map" not in index_value:
            continue
        for name in set(index_value["weight_map"].values()):
            if not (index.parent / name).is_file():
                raise ValueError(
                    f"Missing local role model shard: {index.parent / name}"
                )
    return files


def _snapshot(recipe, *, include_weights, hashes):
    auxiliary, weights = {}, {}
    for logical_path, path in _inventory(recipe).items():
        stat = path.stat()
        if Path(logical_path).suffix in _WEIGHT_SUFFIXES:
            value = {
                "resolved_path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
            if include_weights:
                if path not in hashes:
                    hashes[path] = sha256_file(path)
                value["sha256"] = hashes[path]
            weights[logical_path] = value
        else:
            if path not in hashes:
                hashes[path] = sha256_file(path)
            auxiliary[logical_path] = {
                "resolved_path": str(path),
                "sha256": hashes[path],
            }
    return {"recipe": recipe, "auxiliary_files": auxiliary, "weight_files": weights}


def role_model_resources(models, *, include_weights=False):
    hashes, resources, roles = {}, {}, {}
    for role, model in sorted(models.items()):
        recipe = _recipe(model)
        reference = "local_model_" + digest(recipe) if recipe else None
        roles[role] = {
            "alias": model.alias,
            "model_id": model.model_id,
            "provider": model.provider.value,
            "resource_id": reference,
        }
        if reference is not None and reference not in resources:
            resources[reference] = _snapshot(
                recipe, include_weights=include_weights, hashes=hashes
            )
    return {
        "schema_version": "local_role_resources_v1",
        "roles": roles,
        "resources": resources,
        "weight_verification": "complete" if include_weights else "pending_execute",
    }


def _without_weight_hashes(resource):
    return {
        **resource,
        "weight_files": {
            path: {key: value for key, value in data.items() if key != "sha256"}
            for path, data in resource["weight_files"].items()
        },
    }


def complete_model_resource_binding(binding):
    """Upgrade all preflight role resources; legacy keeps its executor-only key."""
    manifest = binding.get("model_resources")
    if manifest is None:
        if binding.get("settings", {}).get("protocol_version") == "spatialcraft_v2":
            raise ValueError(
                "v2 execution requires preflight identities for every local role model"
            )
        model_path = Path(binding["model_path"])
        shards = sorted(
            set(
                read_json(model_path / "model.safetensors.index.json")[
                    "weight_map"
                ].values()
            )
        )
        binding["weights_sha256"] = {
            name: sha256_file(model_path / name) for name in shards
        }
        return
    hashes, resources = {}, {}
    for reference, prior in manifest["resources"].items():
        current = _snapshot(prior["recipe"], include_weights=True, hashes=hashes)
        if _without_weight_hashes(current) != _without_weight_hashes(prior):
            raise ValueError(
                "Local role model resources changed since preflight; rerun preflight"
            )
        resources[reference] = current
    completed = {**manifest, "resources": resources, "weight_verification": "complete"}
    binding["model_resources"] = completed
    executor = resources[completed["roles"]["executor"]["resource_id"]]
    root = Path(executor["recipe"]["model_path"])
    binding["weights_sha256"] = {
        str(Path(path).relative_to(root)): value["sha256"]
        for path, value in executor["weight_files"].items()
        if Path(path).is_relative_to(root)
    }


def validate_runtime_model_resources(models, binding):
    """No big-weight reread: compare roles, auxiliaries and weight inventory/stats.

    Programmatic test runtimes can be unbound; this is explicitly labelled and
    must not be interpreted as a formally reproducible execution identity.
    """
    manifest = binding.get("model_resources")
    if manifest is None:
        return {"status": "unbound_programmatic_runtime", "role_resources": {}}
    current = role_model_resources(models)
    if current["roles"] != manifest["roles"] or set(current["resources"]) != set(
        manifest["resources"]
    ):
        raise ValueError("Runtime local role model identity differs from preflight")
    for reference, resource in current["resources"].items():
        if resource != _without_weight_hashes(manifest["resources"][reference]):
            raise ValueError("Runtime local role model resources changed after binding")
    return {
        "status": manifest["weight_verification"],
        "role_resources": {
            role: data["resource_id"] for role, data in manifest["roles"].items()
        },
    }


__all__ = [
    "complete_model_resource_binding",
    "resolve_role_models",
    "role_model_resources",
    "validate_runtime_model_resources",
]
