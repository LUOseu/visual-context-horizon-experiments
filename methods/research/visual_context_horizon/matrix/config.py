"""Load, validate and freeze the 8-GPU matrix configuration.

Two suites are understood:

* ``vision-overhead-v1``  (configs/vision_overhead_8gpu.yaml) - the current
  8xA100 study: Qwen3-VL-8B TP=1 and the seven VelaDraft parallel block
  drafters on vLLM 0.22.1.
* ``mdsd-break-even-v1``  (configs/matrix_8gpu.yaml) - the earlier two-target
  suite, still loadable so archived runs can be replayed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

try:
    import yaml
except ImportError:  # the planning CLI reports this; GPU environments pin PyYAML
    yaml = None

DEFAULT_CONFIG = "configs/vision_overhead_8gpu.yaml"
LEGACY_CONFIG = "configs/matrix_8gpu.yaml"

SUITES = ("vision-overhead-v1", "mdsd-break-even-v1")
VISION_SUITE = "vision-overhead-v1"
LEGACY_SUITE = "mdsd-break-even-v1"

REQUIRED_VLLM = {"vision-overhead-v1": "0.22.1", "mdsd-break-even-v1": "0.32.0"}
VISION_TARGET = "qwen3vl_8b"
VISION_DRAFTERS = ("dflash", "dflash2", "dspark", "domino", "jetspec",
                   "dflare", "dfly")
VISION_METHODS = ("ar",) + VISION_DRAFTERS
VISION_EXPERIMENTS = ("exp1_visual", "exp2_visual_kv", "exp3_concurrency",
                      "exp4_profile", "exp5_bonus")
VISION_WAVES = ("wave0_smoke", "wave1_main", "wave2_bonus")
CORE_TASKS = ("textvqa", "docvqa", "chartqa", "vqav2", "coco_long_form",
              "mathvista", "hrbench", "mp_docvqa", "video_mme")
BONUS_TASKS = ("ocrbench", "infovqa")

LEGACY_TARGETS = ("qwen3vl_8b", "qwen36_35b")
LEGACY_METHODS = ("ar", "dflash", "dflash2", "dspark", "dflare")
LEGACY_EXPERIMENTS = ("exp1_phase", "exp2_horizon", "exp3_blocks", "exp3b_tp",
                      "exp4_numerics", "exp5_profile")
LEGACY_WAVES = ("wave0_smoke", "wave1_dense8b", "wave2_moe35b")

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


def repo_root() -> Path:
    override = os.environ.get("VCH_REPO_ROOT")
    if override:
        return Path(override).resolve()
    for parent in Path(__file__).resolve().parents:
        if ((parent / DEFAULT_CONFIG).is_file()
                or (parent / LEGACY_CONFIG).is_file()):
            return parent
    return Path(__file__).resolve().parents[4]


def method_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vendor_root() -> Path:
    """Checkout of LUOseu/VelaDraft that scripts/setup_8gpu.sh installs."""
    value = (os.environ.get("VCH_VELADRAFT_DIR")
             or os.environ.get("VELADRAFT_DIR"))
    if value:
        return Path(value).resolve()
    return repo_root() / "vendor" / "veladraft"


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value

    def substitute(match):
        name, default = match.group(1), match.group(3)
        if name in os.environ:
            return os.environ[name]
        if default is None:
            raise ValueError("environment variable %s is required by the config" % name)
        return default

    return _ENV.sub(substitute, value)


def absolutize(config: dict) -> dict:
    """Resolve hand-editable relative paths against the repository root."""
    root = repo_root()
    for key, value in (config.get("paths") or {}).items():
        path = Path(str(value))
        if not path.is_absolute():
            config["paths"][key] = str((root / path).resolve())
    for target in (config.get("targets") or {}).values():
        value = target.get("path")
        if value and not Path(str(value)).is_absolute():
            target["path"] = str((root / value).resolve())
    return config


def load_config(path=None, validate: bool = True) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the matrix config")
    if path:
        path = Path(path)
    else:
        path = repo_root() / DEFAULT_CONFIG
        if not path.is_file():
            path = repo_root() / LEGACY_CONFIG
    config = _expand(yaml.safe_load(path.read_text(encoding="utf-8")))
    absolutize(config)
    config["_config_path"] = str(path.resolve())
    if validate:
        validate_config(config)
    return config


def validate_config(config: dict) -> None:
    suite = config.get("suite")
    if suite not in SUITES:
        raise ValueError("unexpected suite %r (want one of %s)" % (suite, SUITES))
    environment = config.get("environment") or {}
    want_vllm = REQUIRED_VLLM[suite]
    if environment.get("vllm") != want_vllm:
        raise ValueError("suite %s runs vLLM %s; config says %r"
                         % (suite, want_vllm, environment.get("vllm")))
    for key in ("python", "torch", "transformers", "veladraft_commit"):
        if not environment.get(key):
            raise ValueError("environment.%s is required" % key)
    if suite == VISION_SUITE:
        _validate_vision(config)
    else:
        _validate_legacy(config)


def _validate_vision(config: dict) -> None:
    targets = config.get("targets") or {}
    if set(targets) != {VISION_TARGET}:
        raise ValueError("targets must be exactly ['%s'] for %s"
                         % (VISION_TARGET, VISION_SUITE))
    target = targets[VISION_TARGET]
    for key in ("repo", "revision", "path", "stop_token_ids"):
        if not target.get(key):
            raise ValueError("targets.%s.%s is required" % (VISION_TARGET, key))
    drafters = target.get("drafters") or {}
    if set(drafters) != set(VISION_DRAFTERS):
        raise ValueError("targets.%s.drafters must be exactly %s"
                         % (VISION_TARGET, VISION_DRAFTERS))
    for method, entry in drafters.items():
        for key in ("repo", "revision", "checkpoint", "sha256"):
            if not entry.get(key):
                raise ValueError("targets.%s.drafters.%s.%s is required"
                                 % (VISION_TARGET, method, key))
        if len(entry["sha256"]) != 64:
            raise ValueError("drafter SHA256 must be 64 hex characters: %s" % method)
        adapter = entry.get("adapter")
        if entry.get("runnable", True):
            if not adapter:
                raise ValueError("runnable drafter %s needs an adapter block" % method)
            for key in ("root", "bootstrap", "manifest"):
                if not adapter.get(key):
                    raise ValueError("adapter.%s.%s is required" % (method, key))
        elif not entry.get("blocked"):
            raise ValueError("non-runnable drafter %s must document why" % method)
    matrix = config.get("matrix") or {}
    if tuple(matrix.get("methods") or ()) != VISION_METHODS:
        raise ValueError("matrix.methods must be %s" % (VISION_METHODS,))
    protocol = config.get("protocol") or {}
    if protocol.get("repeats") != 3:
        raise ValueError("the protocol requires three paired repeats")
    widths = protocol.get("block_widths") or []
    mapping = protocol.get("block_width_to_spec_tokens") or {}
    if widths != [4, 8, 16] or [mapping.get(w) for w in widths] != [3, 7, 15]:
        raise ValueError("block widths 4/8/16 must map to 3/7/15 speculative tokens")
    engine = config.get("engine") or {}
    if engine.get("max_model_len") != 24576:
        raise ValueError("matrix max_model_len is frozen at 24576")
    if engine.get("max_new_tokens") != 512:
        raise ValueError("matrix max_new_tokens is frozen at 512")
    if tuple(engine.get("visual_kv_modes") or ()) != ("full", "keep_50",
                                                      "keep_25", "drop_all"):
        raise ValueError("engine.visual_kv_modes must be full/keep_50/keep_25/drop_all")
    for name in VISION_EXPERIMENTS:
        if name not in (config.get("experiments") or {}):
            raise ValueError("experiments.%s is required" % name)
    if set(config.get("waves") or {}) != set(VISION_WAVES):
        raise ValueError("waves must be %s" % (VISION_WAVES,))
    tasks = (config.get("workloads") or {}).get("tasks") or {}
    for task in CORE_TASKS:
        if task not in tasks:
            raise ValueError("workloads.tasks.%s is required" % task)
    for task in BONUS_TASKS:
        if task not in tasks:
            raise ValueError("workloads.tasks.%s is required" % task)


def _validate_legacy(config: dict) -> None:
    targets = config.get("targets") or {}
    if set(targets) != set(LEGACY_TARGETS):
        raise ValueError("targets must be exactly %s" % (LEGACY_TARGETS,))
    for name, target in targets.items():
        for key in ("repo", "revision", "path", "stop_token_ids"):
            if not target.get(key):
                raise ValueError("targets.%s.%s is required" % (name, key))
        if set(target.get("drafters") or {}) != {"dflash", "dflash2", "dspark",
                                                 "dflare"}:
            raise ValueError("targets.%s must pin four symmetric drafters" % name)
        for method, entry in target["drafters"].items():
            for key in ("repo", "revision", "checkpoint", "sha256"):
                if not entry.get(key):
                    raise ValueError("targets.%s.drafters.%s.%s is required"
                                     % (name, method, key))
    if tuple((config.get("matrix") or {}).get("methods") or ()) != LEGACY_METHODS:
        raise ValueError("matrix.methods must be %s" % (LEGACY_METHODS,))
    if (config.get("protocol") or {}).get("repeats") != 3:
        raise ValueError("the protocol requires three paired repeats")
    for name in LEGACY_EXPERIMENTS:
        if name not in (config.get("experiments") or {}):
            raise ValueError("experiments.%s is required" % name)
    if set(config.get("waves") or {}) != set(LEGACY_WAVES):
        raise ValueError("waves must be %s" % (LEGACY_WAVES,))


def is_vision_suite(config: dict) -> bool:
    return config.get("suite") == VISION_SUITE


def paths(config: dict) -> dict:
    base = repo_root()
    out = {}
    for key, value in config["paths"].items():
        path = Path(value)
        out[key] = path if path.is_absolute() else (base / path)
    return out


def experiment(config: dict, name: str) -> dict:
    return dict(config["experiments"][name])


def target(config: dict, name: str) -> dict:
    return config["targets"][name]


def drafter(config: dict, target_name: str, method: str) -> dict:
    return config["targets"][target_name]["drafters"][method]


def adapter(config: dict, target_name: str, method: str) -> dict:
    return config["targets"][target_name]["drafters"][method].get("adapter") or {}


def runnable_methods(config: dict, target_name: str = VISION_TARGET) -> list:
    return [method for method, entry in
            config["targets"][target_name]["drafters"].items()
            if entry.get("runnable", True)]


def blocked_methods(config: dict, target_name: str = VISION_TARGET) -> list:
    return [{"method": method, "reason": entry.get("blocked")}
            for method, entry in config["targets"][target_name]["drafters"].items()
            if not entry.get("runnable", True)]


def task_spec(config: dict, task: str) -> dict:
    return dict(config["workloads"]["tasks"][task])


def task_order(config: dict, bonus: bool = False) -> list:
    base = CORE_TASKS + (BONUS_TASKS if bonus else ())
    known = config["workloads"]["tasks"]
    return [task for task in base if task in known]


def batch_sizes(config: dict) -> list:
    return list(config["experiments"]["exp3_concurrency"]["batch_sizes"])


def spec_tokens(config: dict, block_width: int) -> int:
    return int(config["protocol"]["block_width_to_spec_tokens"][int(block_width)])


def spec_method(config: dict) -> str:
    return config["matrix"]["spec_method"]


def visual_kv_modes(config: dict) -> list:
    return list(config["engine"]["visual_kv_modes"])


def task_budgets(config: dict, task: str, budgets=None) -> list:
    """Experiment budgets intersected with a task-specific restriction."""
    wanted = [int(value) for value in (budgets or config["workloads"]["budgets"])]
    spec = task_spec(config, task)
    allowed = spec.get("budgets")
    if allowed:
        allowed = {int(value) for value in allowed}
        wanted = [value for value in wanted if value in allowed]
    return wanted


def task_frames(config: dict, task: str) -> list:
    frames = task_spec(config, task).get("frames")
    return [int(value) for value in frames] if frames else [1]


def sha256(path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp%d" % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def target_file_pins(config: dict, target_name: str) -> dict:
    """Normalise a target manifest to {filename: sha256}."""
    target = config["targets"][target_name]
    relative = target.get("files_manifest") or target.get("download_manifest")
    if not relative:
        return {}
    manifest = repo_root() / relative
    value = read_json(manifest)
    if value is None:
        raise FileNotFoundError(manifest)
    if "files" in value and isinstance(value["files"], dict):
        return {name: (entry.get("sha256") if isinstance(entry, dict) else entry)
                for name, entry in value["files"].items()}
    return dict(value)


def data_pins() -> dict:
    value = read_json(repo_root() / "configs" / "matrix_data_files.json")
    if value is None:
        raise FileNotFoundError("configs/matrix_data_files.json")
    return value


def freeze(config: dict, extra: dict = None) -> dict:
    """Stable description of everything a run depends on."""
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    frozen = {
        "suite": payload["suite"],
        "version": payload.get("version"),
        "config_sha256": sha256_json(payload),
        "config": payload,
    }
    if extra:
        frozen.update(extra)
    return frozen
