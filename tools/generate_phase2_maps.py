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

if __name__ == "__main__":
    train_dir = REPO_ROOT / "pol_env/Tribes/levels/phase2_pool_1v1/train"
    val_dir = REPO_ROOT / "pol_env/Tribes/levels/phase2_pool_1v1/validation"
    
    print("Generating Phase 2 1v1 training maps...")
    generate_maps(train_dir, count=50, start_seed=10000)
    
    print("\nGenerating Phase 2 1v1 validation maps...")
    generate_maps(val_dir, count=10, start_seed=20000)
    print("\nPhase 2 1v1 map generation complete!")
