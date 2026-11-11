# Your Polytopia AI

This workspace adapts PolyVision for your Apple Silicon Mac. The initial goal is
Bardur economy through turn 10 on 11×11 Drylands maps. Combat and full matches
are future work. It does not connect to the installed Polytopia game.

## Run it

Open a terminal in the `polytopia-ai` folder. Each command uses the isolated
Python environment and automatically selects the project-local Java installation.

```sh
.venv/bin/python local/playground.py build
.venv/bin/python local/playground.py check
.venv/bin/python local/playground.py smoke
```

The smoke run performs 512 action steps, a reduced 100-state interface check,
and saves a model. This verifies the pipeline; it is not enough training to
claim improved play.

Start a longer experiment:

```sh
.venv/bin/python local/playground.py train --steps 500000
```

This uses two simulator processes on CPU, full default 10,000-state interface
validation, and saves checkpoints under `runs/`. Stop with Control-C.
Upstream checkpoints do not preserve full optimizer state for exact resumption.

View learning curves in another terminal:

```sh
.venv/bin/python local/playground.py dashboard
```

Then open http://127.0.0.1:6006.

Compare a checkpoint against scripted/random baselines on five validation maps:

```sh
.venv/bin/python local/playground.py evaluate --model runs/RUN_FOLDER/mac_economy.cleanrl_model
```

Replace RUN_FOLDER with the generated run name. This is an initial diagnostic,
not a statistically conclusive benchmark. Reports go in `outputs/evaluations/`.
Turn-10 stars per turn measures the goal more directly than shaped reward.

## What PPO means

PPO (Proximal Policy Optimization) collects games using the current strategy,
estimates which actions worked better than expected, and updates the network
to favor them. Its clipped objective discourages excessively large updates.
The policy chooses moves; a value estimate helps assess their longer-term results.
Legal-action masks prevent the model from selecting unavailable moves.

## Reproducibility and limitations

- `upstream-commit.txt` records the downloaded source revision.
- `requirements-mac.lock.txt` records installed Python packages. The local
  environment uses a Mac-compatible subset of the upstream Windows-oriented lock.
- Java 11 lives in `.runtime/`; Python packages live in `.venv/`.
- All 5,267 supplied maps are hash-checked. Upstream omits 250 test maps listed
  in its manifest, so the original full split check fails. The local check
  explicitly reports those missing maps and requires all other recorded maps.
- Training and evaluation retain separate map pools.
- Upstream has unresolved repository-wide licensing; its README retains the
  attribution and component license details. This local setup adds no new license
  to upstream code.
