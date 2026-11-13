"""Initialise a fresh PPO agent from an existing checkpoint (`ppo.py --init-from PATH`).

This is a warm start, not a resume: it copies network weights only. The optimizer, learning-rate
schedule and step counter start fresh. Typical use is starting a combat (Phase 2) run from a trained
Phase 1 economy policy.

What is copied (target keys are matched to source keys by name):
- same shape: copied as is;
- action-indexed tensors that grew (the per-id `action_embedding`, a dense actor head, or a composed
  `action_type_embedding`): the source rows are copied into the first rows and the new rows keep their
  fresh initialisation. Allowed only when the target action catalog extends the source catalog, i.e. every
  source global id names the same action in the target (checked by rebuilding both catalogs);
- anything else that differs in shape, or exists on one side only, is left at its fresh initialisation
  and reported.

Reset: `reset="auto"` re-initialises the critic when the environment version differs between the
checkpoint and this run (for example Phase 1 -> combat), because the value of a state is measured in
different rewards there. A comma list of key prefixes resets exactly those; "none" resets nothing.

Combat scalar scaling: combat divides six legacy observation scalars (stars, score, cities, kills, tick,
active tribe; flat indices 6292..6297) by fixed scales, while Phase 1 keeps the raw values. When the two
sides differ, the scale is folded into the copied weights (first-layer columns that read those inputs), or
into the observation-normalisation statistics when the checkpoint uses them, so the copied network computes
the same function of the game state as before.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch

OBS_DIM = 6424
SCALAR_OFFSET = 6292  # 52 spatial channels x 11 x 11; the scalar block starts here
N_LEGACY_SCALARS = 6
# Must match TribesGymWrapper.COMBAT_LEGACY_SCALAR_SCALES (the tick scale is the combat horizon);
# tools/tests/test_init_from.py checks this against register_env.py.
COMBAT_LEGACY_SCALAR_BASE_SCALES = (50.0, 1000.0, 6.0, 10.0, None, 1.0)
SCALED_COMBAT_VERSION_RE = re.compile(r"^combat_v\d+_scaled_scalars_horizon_(\d+)$")
WRAPPER_PREFIX = "policy."  # ObsNormalizedAgent (--norm-obs) nests the network under `policy.`
NORMALIZER_PREFIX = "obs_normalizer."
ACTION_TYPE_TABLE_SUFFIX = "action_type_embedding.weight"
_COMBAT_ONLY_FAMILIES = ("ATTACK", "RECOVER")


class InitFromError(RuntimeError):
    pass


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Return (state_dict, sidecar) for a `.cleanrl_model` checkpoint and its `.action_interface.json`."""
    if not os.path.isfile(path):
        raise InitFromError(f"--init-from checkpoint not found: {path}")
    meta_path = path + ".action_interface.json"
    if not os.path.isfile(meta_path):
        raise InitFromError(
            f"--init-from needs the checkpoint sidecar {meta_path}; checkpoints are two files and must stay together")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise InitFromError(f"{path} does not contain a state_dict")
    return state, meta


def legacy_scalar_scales(environment_version) -> Optional[Tuple[float, ...]]:
    """Scales applied to the six legacy scalars for this environment version, or None if unscaled."""
    m = SCALED_COMBAT_VERSION_RE.match(str(environment_version or ""))
    if not m:
        return None
    horizon = float(int(m.group(1)))
    return tuple(horizon if s is None else float(s) for s in COMBAT_LEGACY_SCALAR_BASE_SCALES)


def scalar_conversion(source_version, target_version) -> Optional[Tuple[float, ...]]:
    """Per-scalar factor c with x_target = c * x_source for the six legacy scalars (None if identical)."""
    src = legacy_scalar_scales(source_version) or (1.0,) * N_LEGACY_SCALARS
    tgt = legacy_scalar_scales(target_version) or (1.0,) * N_LEGACY_SCALARS
    conv = tuple(s / t for s, t in zip(src, tgt))
    return None if all(abs(c - 1.0) < 1e-12 for c in conv) else conv


def catalog_extends(source_n: int, target_n: int, width: int, height: int) -> bool:
    """True if every global id of the `source_n`-sized catalog names the same action in the `target_n` one.

    Rebuilds the environment's GlobalActionCatalog (vocabulary parsed from Types.java, no Java needed) for
    Phase 1 and combat and checks that the larger one only appends families after the smaller one's.
    """
    if int(source_n) == int(target_n):
        return True
    from pol_env.Tribes.py.register_env import GlobalActionCatalog, TribesGymWrapper

    vocab = TribesGymWrapper._load_action_vocab(TribesGymWrapper.__new__(TribesGymWrapper))
    catalogs = {}
    for combat in (False, True):
        cat = GlobalActionCatalog(
            width=int(width), height=int(height),
            tech_types=vocab["TECHNOLOGY"],
            train_unit_types=[u for u in vocab["UNIT"] if u not in ("BOAT", "SHIP", "BATTLESHIP", "SUPERUNIT")],
            resource_types=[r for r in vocab["RESOURCE"] if r in ("ANIMAL", "FISH", "WHALES", "FRUIT", "CROPS", "ORE")],
            building_types=vocab["BUILDING"],
            levelup_choices=vocab["CITY_LEVEL_UP"],
            enable_combat=combat,
        )
        catalogs[int(cat.total_size)] = cat
    src, tgt = catalogs.get(int(source_n)), catalogs.get(int(target_n))
    if src is None or tgt is None:
        return False
    if src.enable_combat or not tgt.enable_combat:
        return False
    shared = {k: v for k, v in src.offsets.items()}
    if any(tgt.offsets.get(k) != v for k, v in shared.items()):
        return False
    # The appended combat families must start at or after the end of the source catalog.
    return all(tgt.offsets[f] >= int(source_n) for f in _COMBAT_ONLY_FAMILIES if f in tgt.offsets)


def _strip_wrapper(key: str) -> str:
    return key[len(WRAPPER_PREFIX):] if key.startswith(WRAPPER_PREFIX) else key


def _resolve_reset(reset: str, source_version, target_version) -> List[str]:
    value = str(reset or "auto").strip()
    if value.lower() == "none":
        return []
    if value.lower() == "auto":
        return ["critic."] if str(source_version) != str(target_version) else []
    return [p.strip() for p in value.split(",") if p.strip()]


def _reads_raw_obs(key: str, tensor: torch.Tensor) -> Optional[int]:
    """Column offset of the legacy scalars if this weight reads raw observations, else None."""
    if tensor.dim() != 2:
        return None
    if tensor.shape[1] == OBS_DIM and key.endswith(".weight"):
        return SCALAR_OFFSET  # MLP first layers (critic.0, actor.0, state_encoder.0)
    if key == "scalar_backbone.0.weight" and tensor.shape[1] == OBS_DIM - SCALAR_OFFSET:
        return 0  # spatial ResNet scalar branch
    return None


def apply_init_from(agent: torch.nn.Module, path: str, *, target_meta: dict, network_arch: str,
                    reset: str = "auto", fold_scalars: bool = True) -> dict:
    """Copy compatible weights from `path` into `agent` in place and return a report.

    `target_meta` is this run's action-interface sidecar (built from the training envs).
    """
    state, source_meta = load_checkpoint(path)

    def check_equal(field, src, tgt):
        if src is not None and tgt is not None and str(src) != str(tgt):
            raise InitFromError(f"--init-from: checkpoint {field}={src} but this run has {field}={tgt}")

    check_equal("network_arch", source_meta.get("network_arch", "mlp"), network_arch)
    for field in ("actor_mode", "observation_dim", "legal_action_feature_dim", "map_width", "map_height",
                  "legal_action_feature_version"):
        check_equal(field, source_meta.get(field), target_meta.get(field))

    source_n = _int_or_none(source_meta.get("action_space_n"))
    target_n = _int_or_none(target_meta.get("action_space_n"))
    width = _int_or_none(target_meta.get("map_width")) or 11
    height = _int_or_none(target_meta.get("map_height")) or 11
    grows = source_n is not None and target_n is not None and source_n != target_n
    if grows:
        if source_n > target_n or not catalog_extends(source_n, target_n, width, height):
            raise InitFromError(
                f"--init-from: checkpoint action catalog ({source_n} ids) is not a prefix of this run's "
                f"({target_n} ids), so action-indexed weights cannot be transferred")

    source_version = source_meta.get("phase1_environment_version")
    target_version = target_meta.get("phase1_environment_version")
    reset_prefixes = _resolve_reset(reset, source_version, target_version)
    conversion = scalar_conversion(source_version, target_version) if fold_scalars else None

    source = {_strip_wrapper(k): v for k, v in state.items()}
    target_state = agent.state_dict()
    source_normalized = any(k.startswith(NORMALIZER_PREFIX) for k in source)
    target_normalized = any(k.startswith(NORMALIZER_PREFIX) for k in target_state)
    if source_normalized != target_normalized:
        raise InitFromError(
            "--init-from: the checkpoint and this run must both use --norm-obs or both not use it "
            f"(checkpoint normalised={source_normalized}, this run normalised={target_normalized})")

    copied, extended, reset_keys, kept_fresh, folded = [], [], [], [], []
    used_source = set()
    with torch.no_grad():
        for tkey, tval in target_state.items():
            key = _strip_wrapper(tkey)
            if any(key.startswith(p) for p in reset_prefixes):
                reset_keys.append(tkey)
                continue
            sval = source.get(key)
            if sval is None:
                kept_fresh.append(tkey)
                continue
            used_source.add(key)
            sval = sval.detach().clone().to(dtype=tval.dtype)
            if conversion is not None:
                if source_normalized and key in (NORMALIZER_PREFIX + "mean", NORMALIZER_PREFIX + "var"):
                    for j, c in enumerate(conversion):
                        sval[SCALAR_OFFSET + j] *= c if key.endswith("mean") else c * c
                    folded.append(tkey)
                elif not source_normalized:
                    col = _reads_raw_obs(key, sval)
                    if col is not None:
                        for j, c in enumerate(conversion):
                            sval[:, col + j] /= c
                        folded.append(tkey)
            if tuple(sval.shape) == tuple(tval.shape):
                tval.copy_(sval)
                copied.append(tkey)
            elif (grows and sval.dim() == tval.dim() and tuple(sval.shape[1:]) == tuple(tval.shape[1:])
                  and sval.shape[0] < tval.shape[0]
                  and ((sval.shape[0], tval.shape[0]) == (source_n, target_n)
                       or key.endswith(ACTION_TYPE_TABLE_SUFFIX))):
                tval[: sval.shape[0]].copy_(sval)
                extended.append(tkey)
            else:
                kept_fresh.append(tkey)
    agent.load_state_dict(target_state)

    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {
        "path": os.path.abspath(path),
        "sha256": sha,
        "source_network_arch": source_meta.get("network_arch", "mlp"),
        "source_environment_version": source_version,
        "target_environment_version": target_version,
        "source_action_space_n": source_n,
        "target_action_space_n": target_n,
        "reset_prefixes": reset_prefixes,
        "scalar_conversion": list(conversion) if conversion else None,
        "copied": len(copied),
        "extended": extended,
        "reset": reset_keys,
        "kept_fresh": kept_fresh,
        "folded": folded,
        "unused_source_keys": sorted(set(source) - used_source),
    }


def format_report(report: dict) -> Iterable[str]:
    yield (f"[INIT_FROM] {report['path']} (sha256 {report['sha256'][:12]}...) "
           f"{report['source_environment_version']} -> {report['target_environment_version']}")
    yield (f"[INIT_FROM] copied={report['copied']} extended={len(report['extended'])} "
           f"reset={len(report['reset'])} kept_fresh={len(report['kept_fresh'])} "
           f"unused_source={len(report['unused_source_keys'])}")
    if report["extended"]:
        yield (f"[INIT_FROM] action ids {report['source_action_space_n']} -> {report['target_action_space_n']}: "
               f"first rows copied for {report['extended']}")
    if report["scalar_conversion"]:
        yield f"[INIT_FROM] folded legacy-scalar scaling {report['scalar_conversion']} into {report['folded']}"
    for name in ("reset", "kept_fresh", "unused_source_keys"):
        if report[name]:
            yield f"[INIT_FROM] {name}: {report[name]}"
