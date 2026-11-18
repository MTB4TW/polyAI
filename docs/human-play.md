# Playing against the agent (`tools/play_vs_agent.py`)

A browser game against a trained combat (Phase 2) checkpoint. It is the same game the agent is evaluated on, with
you in SimpleAgent's seat, so a human result can be compared directly with the agent's win rate against SimpleAgent.

```bash
# On a Mac or any machine with the repo built (local/START-HERE.md):
.venv/bin/python tools/play_vs_agent.py --model-path runs/<run>/<name>.cleanrl_model --player alias1
# then open http://127.0.0.1:8765

# On Farmshare (run it on a compute node or a login node, then forward the port from your laptop):
# on your laptop, to the same login node by name (e.g. rice-03), never on Farmshare itself:
ssh -L 8765:127.0.0.1:8765 <you>@rice-03.stanford.edu     # then open http://127.0.0.1:8765 locally
```

A checkpoint stored in W&B can be fetched with `tools/wandb_checkpoint.py` first. Without `--model-path` the
opponent plays random legal moves. That is only useful for trying out the interface.

## The game

- **The setup is the evaluation setup** (`tools/evaluate_combat.py`):
  - validation maps;
  - capitals mode;
  - the rules, turn limit and fog memory are read from the checkpoint's environment version.
- **Seats:**
  - The agent plays tribe 0 and moves first, as in evaluation.
  - You play tribe 1, the seat SimpleAgent normally takes. You cannot play the first seat yet: every
    checkpoint is trained and evaluated only from that seat (see the roadmap).
- **Fog:** you see only what tribe 1 sees, which is the engine's fogged observation for the active tribe.
  - At the start of your turn, the log lists the changes you can see from the agent's turn. A unit that walks
    into the fog is not reported as killed.
  - The agent's full move list is revealed only when the game ends.
- **Moves:** you choose from every action the engine allows tribe 1, the same set SimpleAgent picks from.
  Star gifts and war declarations are hidden because no combat player uses them.
- **Result:**
  - You win by taking the agent's capital.
  - The agent wins by taking yours.
  - At the turn limit the game is a timeout: the agent did not win.
- **Controls:**
  - Click one of your units or cities. Its actions on the spot (train, capture, harvest, city rewards) appear
    as buttons, and its moves and attacks are highlighted on the board. Click a highlighted tile to play it.
  - Research is in its own panel. <kbd>E</kbd> ends the turn and <kbd>Esc</kbd> clears the selection.
  - "New game" picks a map (random by default) and an optional episode seed.
- **Agent options:**
  - The agent plays argmax (`ppo_argmax`), the policy the evaluation reports.
  - `--sampled` makes it sample its moves instead, which makes it harder to exploit by repeating a line.

## Records

Every finished game is written to `--record-dir` (by default `runs/human_games/`, which is never committed) as
JSON (`human_game_v1`). A record holds:
- the map, the episode seed, the checkpoint and the settings;
- the player alias;
- the agent's outcome;
- every action of both sides, in order.

The agent's moves are stored as global action ids and yours as raw engine actions. The engine is deterministic
given (map, seed, actions), so a record replays exactly:

```bash
.venv/bin/python tools/play_vs_agent.py --verify-record runs/human_games/<file>.json
```

Use an alias with `--player`, not a real name.

## Towards a human protocol

This tool is the first step of Stage 4 in the roadmap. Before reporting "the agent beats humans":
- **Maps:** fix a set of maps in advance.
  - The held-out `eval_test` maps need `--pool eval_test --confirm-test`. Under CLAUDE.md rule 1, use them
    only when a human has asked for that in the session.
  - Never tune on maps a human has played.
- **Players:** record each player's level (for example a Polytopia multiplayer rating) with the alias.
- **Reporting:** report wins with a confidence interval, and say which checkpoint and whether it played argmax
  or sampled.
- **Missing pieces:**
  - The first seat, which needs seat-swap training.
  - Converting records into training data (the human's moves are raw engine actions; the self-play
    relabelling in `combat_selfplay.py` maps them into the agent's action space).
