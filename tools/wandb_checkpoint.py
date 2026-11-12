#!/usr/bin/env python
"""Share PolyVision checkpoints through W&B artifacts.

A checkpoint is two files that must stay together under their original names:
the model (NAME.cleanrl_model) and its action-interface sidecar
(NAME.cleanrl_model.action_interface.json). This tool uploads both as one
artifact, records their SHA-256 hashes, and verifies them on download.

Usage:
    python tools/wandb_checkpoint.py upload runs/experiments/NAME/NAME.cleanrl_model --name NAME
    python tools/wandb_checkpoint.py download NAME:latest --dest runs/experiments/NAME

Project and entity come from --project/--entity, else WANDB_PROJECT/WANDB_ENTITY,
else project "polyAI" under your default W&B entity. Run `wandb login` once first.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

SIDECAR_SUFFIX = ".action_interface.json"
ARTIFACT_TYPE = "polyvision-checkpoint"
DEFAULT_PROJECT = "polyAI"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def project_and_entity(args) -> tuple[str, str | None]:
    project = args.project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    entity = args.entity or os.environ.get("WANDB_ENTITY") or None
    return project, entity


def upload(args) -> None:
    import wandb

    model = Path(args.model_path)
    sidecar = Path(str(model) + SIDECAR_SUFFIX)
    for path in (model, sidecar):
        if not path.is_file():
            raise SystemExit(f"Missing file: {path}")
    metadata = {
        "model_filename": model.name,
        "model_sha256": sha256(model),
        "sidecar_filename": sidecar.name,
        "sidecar_sha256": sha256(sidecar),
    }
    if args.evaluation_id:
        metadata["evaluation_id"] = args.evaluation_id

    project, entity = project_and_entity(args)
    run = wandb.init(project=project, entity=entity, job_type="upload-checkpoint", name=f"upload-{args.name}")
    artifact = wandb.Artifact(args.name, type=ARTIFACT_TYPE, metadata=metadata)
    artifact.add_file(str(model))
    artifact.add_file(str(sidecar))
    logged = run.log_artifact(artifact)
    logged.wait()
    print(f"uploaded {run.entity}/{project}/{args.name}:{logged.version}  model_sha256={metadata['model_sha256']}")
    run.finish()


def download(args) -> None:
    import wandb

    project, entity = project_and_entity(args)
    api = wandb.Api()
    ref = args.artifact if args.artifact.count("/") == 2 else f"{entity or api.default_entity}/{project}/{args.artifact}"
    artifact = api.artifact(ref, type=ARTIFACT_TYPE)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(dest))

    meta = artifact.metadata
    model = dest / meta["model_filename"]
    sidecar = dest / meta["sidecar_filename"]
    for path, expected in ((model, meta["model_sha256"]), (sidecar, meta["sidecar_sha256"])):
        if not path.is_file():
            raise SystemExit(f"Download incomplete, missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"Hash mismatch for {path}: expected {expected}, got {actual}")
    print(f"downloaded {ref} -> {model}  (hashes verified)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project")
    p.add_argument("--entity")
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="upload a checkpoint and its sidecar")
    up.add_argument("model_path")
    up.add_argument("--name", required=True, help="artifact name, e.g. resnet5m-seed1")
    up.add_argument("--evaluation-id", help="optional evaluation folder name to record")
    up.set_defaults(func=upload)

    down = sub.add_parser("download", help="download and verify a checkpoint")
    down.add_argument("artifact", help="NAME:VERSION (e.g. resnet5m-seed1:latest) or entity/project/NAME:VERSION")
    down.add_argument("--dest", required=True, help="folder to download into")
    down.set_defaults(func=download)

    args = p.parse_args()
    try:
        args.func(args)
    except ModuleNotFoundError as e:
        if e.name != "wandb":
            raise
        sys.exit("wandb is not installed in this environment: pip install wandb")


if __name__ == "__main__":
    main()
