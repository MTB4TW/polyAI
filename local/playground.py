"""Local Mac entry point. Run with .venv/bin/python local/playground.py --help."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Polytopia opening-economy AI playground")
    parser.add_argument("task", choices=["build", "check", "smoke", "train", "evaluate", "dashboard"])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--arch", choices=["mlp", "spatial_resnet", "spatial_resnet_local"], default="mlp", help="Neural network architecture")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.steps < 512:
        parser.error("--steps must be at least 512")
    os.chdir(ROOT)
    env = os.environ.copy()
    jdks = sorted((ROOT / ".runtime").glob("jdk-*/Contents/Home"))
    if jdks:
        env["JAVA_HOME"] = str(jdks[-1])
        env["PATH"] = str(jdks[-1] / "bin") + os.pathsep + env["PATH"]
    env.update(POLYVISION_LEVEL_POOL_GLOB="levels/phase1_pool_bardur_real/train/*.csv",
               POLYVISION_SOLO_NO_OPPONENT_MODE="1", POLYVISION_INFO_MODE="fast",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    def run(command, cwd=ROOT):
        subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True)

    python = sys.executable
    if args.task == "build":
        engine = ROOT / "pol_env/Tribes"
        sources = sorted((engine / "src").rglob("*.java"))
        run(["javac", "-cp", "lib/json.jar", "-d", "out", "-sourcepath", "src", *sources], engine)
    elif args.task == "check":
        pool = ROOT / "pol_env/Tribes/levels/phase1_pool_bardur_real"
        manifest = json.loads((pool / "split_manifest.json").read_text())
        missing_test = 0
        checked = 0
        for entry in manifest["maps"]:
            path = pool / entry["relative_path"]
            if entry["pool"] == "test" and not path.exists():
                missing_test += 1
                continue
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["csv_sha256"]:
                raise RuntimeError(f"Map missing or hash mismatch: {path}")
            checked += 1
        print(f"Verified {checked} map hashes; {missing_test} test maps unavailable upstream.", flush=True)
        run([python, "tools/validate_environment_contract.py", "--expected-width", "11",
             "--expected-height", "11", "--max-maps", "1"])
    elif args.task in ("smoke", "train"):
        smoke = args.task == "smoke"
        run([python, "py_rl/cleanrl/cleanrl/ppo.py", "--total-timesteps",
             "512" if smoke else str(args.steps), "--num-envs", "1" if smoke else "2",
             "--num-steps", "128", "--no-cuda", "--save-model",
             "--network-arch", args.arch,
             "--exp-name", "mac_smoke" if smoke else f"mac_economy_{args.arch}",
             "--validation-states", "100" if smoke else "10000"])
    elif args.task == "evaluate":
        if args.model is None:
            parser.error("evaluate requires --model pointing to a .cleanrl_model checkpoint")
        run([python, "tools/evaluate_phase1.py", "--model-path", args.model.resolve(),
             "--pool", "validation", "--max-maps", "5", "--repeats-per-map", "1"])
    else:
        run([python, "-m", "tensorboard.main", "--logdir", "runs", "--host", "127.0.0.1"])


if __name__ == "__main__":
    main()
