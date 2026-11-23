# Handoff: combat league work, 2026-11-17 → 2026-11-23

This is for picking the work up in a new chat without the old context. It says what exists, where it is, what was
learned, and what is open. Details and every number are in `docs/experiments.md` (newest first) and `CHANGELOG.md`
(entries 085–093). Kevin's Claude agent did this work, on Farmshare under `kevlai`.

## 1. The branch and PR stack

All work is pushed. The stack is based on Devin's league branch, which is not on `main`, so the PRs must merge in
order:

| PR | Branch | Base | What |
|---|---|---|---|
| #32 | `feat/combat-unit-status` | `claude/combat-critic-init-96179a` (Devin) | HP / veteran / has-acted planes and attack-outcome action features (CHANGELOG 087) |
| #33 | `exp/league-unitstatus-wikimaps` | #32 | Map pool versions + the wiki-parity pool (088), league opponents' village-capture fix (089), stop/resume and map-generation commits |
| #34 | `feat/combat-optional-rules` | #33 | Optional wrapper rules and extra engine actions (090), road / Riding rewards (091), head-to-head fixes, every experiment record of the week |
| #37 | `feat/play-current-models` | #34 | The play tool on this code (092, 093) and this file. **Top of the stack: check this out to have everything** |
| #29 | `feat/play-vs-agent` | `main` | Earlier play tool on main; #37 supersedes it for the current checkpoints |
| #30 | `feat/resumable-training` | `main` | Stop/resume (085) for main (also cherry-picked into #33) |
| #31 | `fix/map-generation-wiki` | `main` | Map generator matched to the wiki (086) for main (also cherry-picked into #33) |
| #28 | `feat/game-store` | `main` | Saved-game engine (older) |

- Merge order for the stack: Devin's `claude/combat-critic-init-96179a`, then #32, #33, #34, #37.
- CHANGELOG entry numbers in these branches (085–093) may collide with entries merged to `main` meanwhile. Renumber
  on merge.

Local worktrees (Farmshare) are under `/home/users/kevlai/polyAI/.claude/worktrees/`:
- `unitstatus`, `league-us`, `rules`, `play`, `resume`, `mapgen`.
- Each has its own compiled Java `pol_env/Tribes/out/`, built with `/home/users/kevlai/polyAI/.runtime/jdk-11.0.22+7`.
- `.venv` and `.runtime` are symlinks to the main checkout's.
- Training runs live in each worktree's `runs/` (not in git).

## 2. What the code can do now (all opt-in; defaults unchanged)

| Feature | Switch | CHANGELOG |
|---|---|---|
| Unit HP / veteran / has-acted inputs + attack outcome features | `POLYVISION_UNIT_STATUS=1` | 087 |
| Map pool version | `POLYVISION_COMBAT_POOL=phase2_pool_1v1_wiki` (wiki-parity maps; default `phase2_pool_1v1`) | 088 |
| League snapshots make the deferred village captures (bug fix, always on) | — | 089 |
| Switch off the wrapper's hard-coded rules (forced village moves, auto capture, unit-2 backtrack, city wall / fishing masks) | `POLYVISION_COMBAT_RULES_OFF=all` or names | 090 |
| Offer engine actions the wrapper never offered (BUILD_ROAD, MAKE_VETERAN, DISBAND, BURN_FOREST, …) | `POLYVISION_COMBAT_EXTRA_ACTIONS=all` or names | 090 |
| League snapshots keep their own trained rules in a share of games | `POLYVISION_COMBAT_LEAGUE_OWN_RULES_PROB=0.5` | 090 |
| Roads / Riding research bonuses; exempt techs from `unused_tech` | `POLYVISION_COMBAT_REWARD_ROADS_TECH`, `_RIDING_TECH`, `POLYVISION_COMBAT_UNUSED_TECH_EXEMPT=RIDING,ROADS` | 091 |
| Stop and resume runs exactly; resubmits itself at the 48 h limit | `touch runs/<run>/STOP`, `RESUME_FROM=<run dir>`, `AUTO_RESUME=1` | 085 |
| Head-to-head between checkpoints that differ in unit status / rules / extra actions | `tools/evaluate_combat_h2h.py` (B plays under A's settings) | 090 |

The launch script for all league runs is `cluster/league_unit_status_wiki.sbatch` (in #33/#34):
- `INIT_CHECKPOINT` sets the warm start.
- `OVERRIDE_ENV="K=V …"` sets reward weights. It is applied after the shared env files, which set the weights
  unconditionally.
- `EXTRA_PPO_ARGS_APPEND` adds trainer arguments.
- `WANDB_MODE=offline` is used because of Farmshare DNS (§6).

## 3. The models

| Model | What | Where |
|---|---|---|
| **Riders final** (`league_us_wiki_norules_riders_4m`) | No wrapper rules, all actions, Riding/Rider/Road rewards; 8M steps of lineage (no-rules 4.0M + 4M). **Best so far** | `.claude/worktrees/rules/runs/progress_ckpts/riders_final_4m.cleanrl_model`; W&B artifact `league_us_wiki_norules_riders_4m-final` |
| No-rules final (`league_us_wiki_norules_8m`) | No wrapper rules, all actions; 8M from Devin's league8m seg1 @1.2M | `…/progress_ckpts/norules_final_8m.cleanrl_model`; W&B `league_us_wiki_norules_8m-final` |
| Rules-on best (`league_us_wiki_8m` @2.0M) | Wrapper rules on; flat after 2.0M, stopped at 5.9M | `…/progress_ckpts/ruleson_2000000.cleanrl_model`; W&B `league_us_wiki_8m-best-2m` |
| Sweep finals (`sweep_*`, 2M each from the riders final) | Overnight 2026-11-22 sweep, §4 | `.claude/worktrees/rules/runs/sweep_eval_ckpts/*_final.cleanrl_model` |
| Devin's league8m seg1–4 (to 12M) | Rules on, old map pool, no unit status | `/home/users/devinlai/polyAI-wt/league3/runs/…`; copies in `.claude/worktrees/play/runs/play_models_older/` |

W&B (team `polyai-stanford`, project `polyAI`, group `league-2026-11`) has clean runs uploaded from the TensorBoard
logs:
- `lwfxh78t` is rules on, `g69cysyd` is no rules, `2nm9j5r5` is riders.
- Older pages `8gn7ac6q`, `e9trzjkz`, `w4t9bxaa`, `9pm04r8c` and `a1smxuqs` have duplicated or cut history. They
  can be deleted (not done).

## 4. What was learned (numbers in `docs/experiments.md`)

1. **League opponents could never take a village (fixed, 089).**
   - The deferred village capture ran only in the learner's `step()`, so every league snapshot played crippled.
   - That inflated league win rates to about 0.8 against copies of the learner.
   - It also made "first mover wins 94%" in head-to-heads. The true first-move edge is about 0.68 with sampling.
2. **The wrapper's scripted rules are a crutch.**
   - A model trained with them is about 8 points weaker without them.
   - Trained without them (no-rules run), it catches up.
   - The Riding / Rider / Road rewards made a branch that beat the plain no-rules run at equal training (0.585). By
     8M the plain run caught up (final vs final 0.519, even).
3. **Best model vs Devin's newest (12M): 0.745** (61% win / 27% timeout / 12% loss).
4. **SimpleAgent is saturated.** Every current checkpoint wins 94–100%.
5. **Timeouts are the main weakness.**
   - 35–54% of games between strong models end at turn 20.
   - Playing argmax instead of sampling does not reduce them, so sampling noise is not the cause.
   - Armies are about 85–90% warriors (riders 0.1–0.4, giants about 1 per game).
   - About 17 kills each side per game, 3 cities each, and roads are essentially never built (0–0.03 per game).
   - Bonuses get gamed: Riding is researched late (about turn 15) to collect the reward without changing the army.
6. **Rules-on run plateaued at 2.0M** (4.8M vs 2.0M is even). The no-rules run kept improving to 3.2M+.
7. **Overnight sweep (from the riders final, 2M each)**, varying the timeout penalty (−13 / −14), seed, learning rate
   1e-4, entropy 0.02, city reward 2, and the no-rules final as init:
   - At 1.0M, none beat the unchanged control. The timeout penalty did not reduce timeouts.
   - LR 1e-4 and entropy 0.02 were worse.
   - Final (2M) results: see `docs/experiments.md`, section "Overnight sweep … final".

**So the next lever is what the agent can do (economy, army mix, finishing a siege), not reward bonuses.** Ideas
not yet tried:
- opponents in the league that punish warrior spam (a scripted rider rush);
- potential-based rewards on stars per turn or army value instead of one-off bonuses;
- a time-gated Riding bonus;
- a longer horizon (30 turns; a contract change).

## 5. Playing the models (browser)

```bash
cd /home/users/kevlai/polyAI/.claude/worktrees/play
export PATH=/home/users/kevlai/polyAI/.runtime/jdk-11.0.22+7/bin:$PATH
nohup .venv/bin/python tools/play_vs_agent.py --player human1 --only-models \
  --models 'runs/play_models/*.cleanrl_model' --models 'runs/play_models_older/*.cleanrl_model' \
  --map-pool phase2_pool_1v1_wiki --model-path runs/play_models/no_rules_riders_final.cleanrl_model \
  > logs/play_server.log 2>&1 &
# laptop:  kinit kevlai@stanford.edu
#          ssh -o GSSAPIAuthentication=yes -L 8765:127.0.0.1:8765 kevlai@rice-0N.farmshare.stanford.edu
# browser: http://127.0.0.1:8765
```

- 12 labelled opponents come from `runs/play_models/labels.json` and `runs/play_models_older/labels.json`.
- You can choose your tribe and the agent's: a pool map is used if one has the pairing, otherwise a map is
  generated.
- Games are saved in `runs/human_games/`, with replays.
- The server keeps one game at a time. A second person should run their own server on another `--port`.
- See `docs/human-play.md`. Devin can read the model folders directly.

## 6. Operational notes (learned the hard way)

- **SLURM on login nodes.** `sackd` failed on rice-03 (Nov 20) and on rice-04 (Nov 22), so `squeue` and `sbatch` fail
  there. Use another login node; the jobs themselves are unaffected. Claude cannot ssh between login nodes, because it
  has no Kerberos ticket. The user must start the session on a working node (`claude --continue` resumes the chat).
- **W&B DNS.** `api.forge.coreweave.com` intermittently fails DNSSEC validation on Farmshare. Train with
  `WANDB_MODE=offline`.
- **W&B uploads.**
  - `wandb beta sync --live` crashes with "block appears to be zeroed", and restarting it duplicates the history.
  - After a run ends, upload with `.claude/worktrees/rules/runs/tools_scratch/tb_to_wandb.py RUN_DIR NAME --group G`.
    It reads the TensorBoard logs into one clean run.
- **`/tmp` and the scratchpad are per login node** and get cleaned. Keep scripts in `runs/tools_scratch/`.
- **`league_stats.json` resets on every resume.** Per-opponent history is in TensorBoard (`league/games/*`,
  `league/win_rate/*`, last ~1000 games). Recover it with `runs/tools_scratch/tb_league_file.py EVENT_FILE`.
- **Reading the TensorBoard logs is slow** (0.3–0.9 GB per run, 10–30 min). The readers in `runs/tools_scratch/` are
  `tb_last.py`, `tb_final.py`, `tb_wtl_series.py` and `tb_league_file.py`. Run them in the background.
- **League globs match by file name.** `LEAGUE_GLOB` / `POLYVISION_COMBAT_LEAGUE_GLOB` pick up anything matching:
  - The riders branch's `progress_ckpts/norules_*` glob also took in a checkpoint copied there for a match.
  - The sweep's league includes `progress_ckpts/*.cleanrl_model` on purpose.
- **Fair-share.** 8 × 32-CPU runs pushed small evaluation jobs behind other users' for hours. `bigmem` is not
  available to this account.
- **Rules for evaluation and training:**
  - League win rates count timeouts as non-wins, and the league over-samples hard opponents.
  - Judge strength by head-to-head (`tools/evaluate_combat_h2h.py`, 480 games: `--repeats-per-map 2`, timeout = ½).
- **Never evaluate `eval_test`** unless a human asks (CLAUDE.md).

## 7. Open threads

- Merge the PR stack in order (§1) and renumber CHANGELOG entries.
- The engine flag `GameState.pushedUnitsStartFresh` (093) is on only in the play tool. Making it the training
  default is a rules revision (`polytopia_2_15` r3) that would invalidate current checkpoints. That is a decision to
  make.
- The no-rules agents sometimes choose DECLARE_WAR, which is harmless but useless. Consider dropping DECLARE_WAR and
  SEND_STARS from `POLYVISION_COMBAT_EXTRA_ACTIONS`.
- Timeouts / economy / army mix (§4) is the research question for the next runs.
