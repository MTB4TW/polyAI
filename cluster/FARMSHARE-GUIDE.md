# Stanford Farmshare GPU Cluster Guide for Polytopia AI

This guide walks through deploying, training, and monitoring large-scale Polytopia AI experiments on the **Stanford Farmshare** high-performance compute cluster using SLURM.

---

## 1. Connect to Stanford Farmshare

Log into Farmshare using your Stanford SUNet ID:

```bash
ssh <sunetid>@rice.stanford.edu
# Or alternatively:
# ssh <sunetid>@wheat.stanford.edu
```

---

## 2. Sync the Codebase to Farmshare

From your local Mac terminal (inside the `polytopia-ai` folder), sync the repository to your Farmshare home directory:

```bash
# Run this from your local Mac:
rsync -avz --exclude='.venv' --exclude='runs' --exclude='.git' \
    ./ <sunetid>@rice.stanford.edu:~/polytopia-ai/
```

*(Alternatively, you can push your changes to a private GitHub repo and `git clone` on Farmshare).*

---

## 3. One-Step Environment Setup on Farmshare

On Farmshare (via SSH), navigate into the folder and run the automated setup script:

```bash
cd ~/polytopia-ai
bash cluster/setup_farmshare.sh
```

This script:
1. Loads cluster modules (`cuda/12.1`, `java/11`, `python/3.11`).
2. Creates an isolated Linux virtual environment (`.venv`).
3. Installs CUDA-enabled PyTorch, Gymnasium, and dependencies from `cluster/requirements-linux.txt`.
4. Compiles the Java Tribes game engine.
5. Runs a pre-flight test verifying geometry and map hashes.

---

## 4. Launching Training via SLURM

### Option A: Standard 5 Million Step GPU Run
Submit the batch script:

```bash
sbatch cluster/train_farmshare.sbatch
```

The job configuration in `cluster/train_farmshare.sbatch`:
* **Partition**: `gpu` with 1 NVIDIA GPU.
* **CPUs**: 16 dedicated CPU cores.
* **Parallel Game Envs**: 16 concurrent JVM simulation processes running rollouts simultaneously.
* **Expected Throughput**: ~1,500 to 2,500 steps/second on cluster hardware (5M steps finishes in ~35–50 minutes; 20M steps in ~2–3 hours).

### Option B: Customizing Steps or Environments
You can pass environment variables directly to `sbatch`:

```bash
# 10 Million steps with 24 parallel environments
NUM_ENVS=24 TOTAL_TIMESTEPS=10000000 EXP_NAME=farmshare_10m sbatch cluster/train_farmshare.sbatch
```

---

## 5. Monitoring Jobs and Learning Curves

### Check Job Status
```bash
squeue -u $USER
```

### View Live Standard Output & Logs
```bash
tail -f logs/polytopia_ai-*.out
```

### Option 1: Live TensorBoard via SSH Tunnel
On Farmshare, launch TensorBoard in a screen/tmux session:
```bash
source .venv/bin/activate
tensorboard --logdir runs --port 6006 --host 127.0.0.1
```

From your local Mac terminal, open the SSH tunnel:
```bash
ssh -N -L 6006:localhost:6006 <sunetid>@rice.stanford.edu
```
Then visit **`http://localhost:6006`** in your Mac browser.

### Option 2: Remote Tracking with Weights & Biases (W&B)
If you have a [W&B account](https://wandb.ai):
```bash
wandb login
# In train_farmshare.sbatch, add the flags:
# --track --wandb-project-name polytopia-ai
```
This streams live learning curves directly to your online dashboard.

---

## 6. Copying Checkpoints Back to Local Mac

Once training completes on Farmshare, checkpoints are saved under `~/polytopia-ai/runs/`. Download the trained model to your Mac:

```bash
# Run this on your Mac:
rsync -avz <sunetid>@rice.stanford.edu:~/polytopia-ai/runs/ ./runs/
```

Then evaluate your newly trained model against the human benchmarks on your Mac:

```bash
.venv/bin/python tools/evaluate_superhuman.py \
    --model-path runs/farmshare_5m/mac_economy.cleanrl_model \
    --pool human_benchmark \
    --max-maps 17 \
    --output outputs/evaluations/human-diagnostic.json
```

## Training corrections (2026-10-23)

Rebuild the Java engine after syncing these changes (rerun setup). The bridge
now advances the combat round clock; without this fix, income reset to the
starting balance each round.

Combat uses terminal win +1 / loss -1 rewards, zero intermediate reward, and
zero for a timeout at `POLYVISION_MAX_TURNS` completed player turns (default 20).
The horizon is part of the checkpoint environment version. Economy shaping
remains unchanged for Phase 1. Combat dashboards expose win/loss/timeout rates;
these rates are per completed batch of episodes, not cumulative estimates.
Sparse rewards may require a longer horizon or an easier opponent curriculum;
measure timeout frequency before committing to a long training run.

`NETWORK_ARCH=spatial_resnet_local` selects the experimental source/target
spatial-feature architecture. Existing `spatial_resnet` checkpoints still load
unchanged. The local variant requires a new training run and `legal_features`.
`NETWORK_ARCH=spatial_resnet_composed` goes further and drops the per-action-ID
embedding table (about 93% of the ResNet's parameters): actions are scored from
an action-type embedding, the grid features at their tiles, and the legal-action
features. It also requires a new training run and `legal_features`.
Compare both architectures at equal budgets on validation maps. Do not tune
against the held-out test pool or historical human reference scores.

Use `tools/evaluate_phase1.py` for canonical evaluations; it now reads the
architecture from checkpoint metadata. `tools/evaluate_superhuman.py` is a
noncanonical diagnostic despite its historical filename. It requires an exact
checkpoint and a new output path, reports the heuristic honestly (no simulated
lookahead), and reads matching map/seed human attempt records. Historical human
records do not establish current environment parity or superhuman performance.

Combat observations retain the 6,424-value size but use signed ownership:
friendly unit-type cells are +1, enemy unit-type cells are -1; enemy territory
is -1 in the first territory plane, while friendly city slots remain +1.
Fog still zeros these planes. The combat environment version is separate from
Phase 1, so old economy and earlier combat checkpoints are rejected as combat
models. Combat action features still lack explicit damage and health estimates;
this remains a training baseline, not a complete competitive battle agent.

## Combat job failure and reset fix (2026-10-30)

Jobs 1875722 (`combat_resnet_2m`) and 1875724 (`combat_local_2m`)
failed at 240 global environment steps, before the first 2,048-step PPO update.
Both passed preflight, then tried to select an action from a finished game's
empty legal mask. No trained checkpoints were saved by either run.

Gymnasium 1.3 defaults to NEXT_STEP autoreset. The trainer now explicitly uses
SAME_STEP so the next policy action receives the reset game's observation and
legal actions. Terminal statistics remain separate in `final_info`, with its
nested vector format normalized for episode and combat logging. The empty-mask
check remains enabled: invalid nonterminal states must still fail loudly.

Sync the updated `py_rl/cleanrl/cleanrl/ppo.py` before resubmission. This fix
requires no Java rebuild. Startup now prints `autoreset=SameStep` and the
Gymnasium version. Use unique experiment names for reruns.

When copying combat runs back, omit the trailing slash after the wildcard so
run directories remain separate (replace YOUR_SUNET with your username):

```bash
rsync -avz 'YOUR_SUNET@rice.stanford.edu:~/polytopia-ai/runs/*combat*' runs/
```
