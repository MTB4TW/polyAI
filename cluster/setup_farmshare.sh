#!/usr/bin/env bash
# Setup script for Stanford Farmshare (SLURM GPU cluster)
set -euo pipefail

echo "=========================================="
echo "Polytopia AI: Stanford Farmshare Setup"
echo "=========================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 1. Inspect environment modules (non-blocking)
if command -v module &> /dev/null; then
    echo "Checking available modules..."
    module load cuda 2>/dev/null || true
    module load openjdk 2>/dev/null || module load java 2>/dev/null || true
    module load python3 2>/dev/null || module load python 2>/dev/null || true
fi

# 2. Check Java / JDK availability, auto-installing portable JDK 11 into .runtime if needed
mkdir -p "$ROOT_DIR/.runtime"
if ! command -v javac &> /dev/null; then
    LOCAL_JDK_BIN=$(find "$ROOT_DIR/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
    if [ -n "$LOCAL_JDK_BIN" ]; then
        echo "Found local JDK in .runtime: $LOCAL_JDK_BIN"
        export PATH="$LOCAL_JDK_BIN:$PATH"
        export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
    else
        echo "System javac not found. Downloading portable OpenJDK 11 (Linux x64) into .runtime/..."
        curl -sL "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.22%2B7/OpenJDK11U-jdk_x64_linux_hotspot_11.0.22_7.tar.gz" | tar -xz -C "$ROOT_DIR/.runtime/"
        LOCAL_JDK_BIN=$(find "$ROOT_DIR/.runtime" -maxdepth 3 -name "javac" -exec dirname {} \; | head -n 1)
        export PATH="$LOCAL_JDK_BIN:$PATH"
        export JAVA_HOME="$(dirname "$LOCAL_JDK_BIN")"
    fi
fi
echo "Java compiler: $(javac -version 2>&1)"
echo "Java runtime: $(java -version 2>&1 | head -n 1)"

# 3. Check Python availability
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Python found: $($PYTHON_BIN --version)"

# 4. Setup Python virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    $PYTHON_BIN -m venv "$VENV_DIR" || virtualenv -p "$PYTHON_BIN" "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r cluster/requirements-linux.txt

# 5. Verify PyTorch installation
python -c '
import torch
print("PyTorch version:", torch.__version__)
print("CUDA runtime available in PyTorch:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("Note: Login node (rice-04) is CPU-only. GPUs are attached when running under SLURM.")
'

# 6. Compile Java Tribes engine
echo "Compiling Java Tribes engine..."
cd pol_env/Tribes
mkdir -p out
javac -cp "lib/json.jar" -d out -sourcepath src $(find src -name "*.java")
cd ../..

# 7. Validate environment contract
echo "Validating environment contract..."
export POLYVISION_LEVEL_POOL_GLOB="levels/phase1_pool_bardur_real/train/*.csv"
export POLYVISION_SOLO_NO_OPPONENT_MODE="1"
export POLYVISION_INFO_MODE="fast"
python tools/validate_environment_contract.py --expected-width 11 --expected-height 11 --max-maps 1

echo "=========================================="
echo "Farmshare setup completed successfully!"
echo "You can now submit training jobs via SLURM:"
echo "  sbatch cluster/train_farmshare.sbatch"
echo "=========================================="
