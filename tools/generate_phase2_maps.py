#!/usr/bin/env python
"""Generate 2-Player 11x11 Drylands Maps for Phase 2 Combat Training."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIBES_DIR = REPO_ROOT / "pol_env/Tribes"

# Ensure local Java is in PATH
jdks = sorted((REPO_ROOT / ".runtime").glob("jdk-*/Contents/Home"))
if jdks:
    os.environ["JAVA_HOME"] = str(jdks[-1])
    os.environ["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ.get("PATH", "")

def generate_maps(output_dir: Path, count: int, start_seed: int = 1000):
    output_dir.mkdir(parents=True, exist_ok=True)
    java_bin = "java"
    if jdks:
        java_bin = str(jdks[-1] / "bin/java")
    
    classpath = f"out:lib/json.jar"
    tribes_pairs = [
        ("BARDUR", "IMPERIUS"),
        ("BARDUR", "XIN_XI"),
        ("BARDUR", "OUMAJI"),
    ]
    
    for i in range(count):
        seed = start_seed + i
        t1, t2 = tribes_pairs[i % len(tribes_pairs)]
        filename = f"map_2p_{seed:06d}.csv"
        out_file = output_dir / filename
        
        cmd = [
            java_bin,
            "-cp", classpath,
            "core.levelgen.GenerateLevelCli",
            str(seed),
            "11",
            str(out_file.resolve()),
            "1.0",
            "DRYLANDS",
            t1,
            t2
        ]
        
        res = subprocess.run(cmd, cwd=TRIBES_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Error generating map {filename}: {res.stderr}")
            sys.exit(1)
        print(f"Generated {filename} ({t1} vs {t2})")

POOL_ROOT = REPO_ROOT / "pol_env/Tribes/levels/phase2_pool_1v1"

# Combat evaluation pools (tools/evaluate_combat.py). Seeds do not overlap train (10000+) or validation (20000+).
# 120 maps each, so every tribe pairing appears 40 times. eval_test is held out: evaluate it only when a person
# explicitly asks (CLAUDE.md rule 1).
EVAL_POOLS = {"eval_validation": (120, 30000), "eval_test": (120, 40000)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", help="Generate one pool: train, validation, or a combat evaluation pool "
                                       f"({', '.join(EVAL_POOLS)}); default regenerates train and validation")
    parser.add_argument("--count", type=int, help="Number of maps (default: the pool's standard size)")
    parser.add_argument("--start-seed", type=int, help="First map seed (default: the pool's standard seed)")
    args = parser.parse_args()
    standard = {"train": (50, 10000), "validation": (10, 20000), **EVAL_POOLS}
    if args.pool is None:
        print("Generating Phase 2 1v1 training maps...")
        generate_maps(POOL_ROOT / "train", count=50, start_seed=10000)
        print("\nGenerating Phase 2 1v1 validation maps...")
        generate_maps(POOL_ROOT / "validation", count=10, start_seed=20000)
        print("\nPhase 2 1v1 map generation complete!")
    else:
        if args.pool not in standard and (args.count is None or args.start_seed is None):
            raise SystemExit(f"Unknown pool {args.pool!r}: give --count and --start-seed")
        count, seed = standard.get(args.pool, (None, None))
        generate_maps(POOL_ROOT / args.pool, count=args.count or count, start_seed=args.start_seed or seed)
