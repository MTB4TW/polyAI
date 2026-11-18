# Playing against the agent (`tools/play_vs_agent.py`)

A browser game against a trained combat (Phase 2) checkpoint, followed by a step-by-step replay.

```bash
# On Farmshare (or any machine with the repo built). Devin's runs are listed with --models:
.venv/bin/python tools/play_vs_agent.py --player alias1 \
  --models '/home/users/devinlai/polyAI-wt/*/runs/*/*.cleanrl_model'

# On your laptop, to the same login node by name (e.g. rice-03), never on Farmshare itself:
ssh -L 8765:127.0.0.1:8765 <you>@rice-03.stanford.edu     # then open http://127.0.0.1:8765 locally
```

## Choosing the opponent

The "New game" panel lists every combat checkpoint found in this checkout's `runs/` plus the `--models` globs.
You can also set `POLYAI_MODEL_GLOBS`, with globs separated by `:`.
- **Listing:** newest first and grouped by run, by training step (or "final"), save time and canonical
  evaluation where this checkout has one.
- **Default:** the newest file, or the one given with `--model-path`.

The banner above the board always states which weights you are playing:
- the run and training step;
- the file name and save time, and the first 12 characters of the file's SHA-256;
- the network and the environment version;
- the checkpoint it was warm-started from;
- its canonical evaluation (against SimpleAgent, with the agent in seat 0);
- whether it plays argmax or sampled.

Records store the full SHA-256. "Random legal moves" is there for trying out the interface.

## The game

The setup is the evaluation setup (`tools/evaluate_combat.py`): validation maps, capitals mode, and the rules,
turn limit and fog memory read from the checkpoint. You win by taking the agent's capital, and it wins by
taking yours. At the turn limit the game is a timeout.

| Your seat | You | The agent |
|---|---|---|
| **Second** (tribe 1, default) | Any action the engine allows tribe 1, the same set SimpleAgent chooses from in evaluation. Only star gifts and war declarations are hidden. | Tribe 0 through the training wrapper, exactly as in evaluation. |
| **First** (tribe 0) | The agent's own legal-action list for that seat, through the wrapper. A few engine actions the agent never sees are not offered. | Tribe 1, the way league snapshots play it (`combat_selfplay.OpponentView`): its own fogged view, relabelled so it appears as tribe 0. Its turn count is the turns it has finished, as in seat-swap training. |

- **Which checkpoints know seat 1:** seat-swap league runs (CHANGELOG 072 on `claude/combat-critic-init-96179a`,
  e.g. `combat_pcritic_t1_league_swap_4m`) trained in both seats. Older checkpoints only ever played seat 0,
  so playing first tests them in a seat they never trained in.
- **Fog:** you see only your tribe's fogged observation. At the start of your turn, the log lists the changes
  you can see from the agent's turn.
- **On the board:**
  - resources (🍒🦌🐟🐋🌾🪨🏛️) and buildings, with monuments ringed in gold;
  - lighthouses (🗼, faint until you have found them). As in the game, a lighthouse you have found also shows
    whether the agent has found it (coloured dots);
  - HP bars against true max HP, with the number on each unit.
  - Click any tile to inspect it, including the exact HP of an enemy unit you can see.
- **The agent's turn:** its moves are animated one at a time at the start of your turn, as far as you can
  see them. Choose instant, fast, normal or slow; Skip; Show again. Attacks are a red ⚔️ arrow from attacker
  to target, with damage numbers.
- **Controls:**
  - Click one of your units or cities. Actions on the spot (train, capture, harvest, city rewards) appear as
    buttons, and moves and attacks are highlighted on the board. Click a highlighted tile to play it.
  - Research has its own panel. <kbd>E</kbd> ends the turn and <kbd>Esc</kbd> clears the selection.

## The replay

When the game ends, **Watch the replay** opens it in the combat replay viewer (`tools/combat_replay_template.html`,
the same viewer as `tools/export_combat_replay.py`):
- **Steps:** one step per action of either side, with a slider, previous and next step or turn, and play.
- **Moves on the board:** arrows for moves, outlines for damage and captures, and crosses for kills.
- **Views:** **True board**, **Your view** and **Agent's view**. <kbd>F</kbd> cycles through them.
  - The true board marks the agent's pieces you could not see.
  - A fogged view hides pieces and events inside that player's fog.
- **Side panels:** a game log listing every move of both players, and per-step events and stats.

Each player's view is rebuilt from the true board and the tiles that player had explored at that step. This is
exactly how the engine fogs an observation: every actor on an explored tile is visible. A tribe's explored area
only grows during its own turn.

The viewer is shared: `export_combat_replay.py` replays (agent against SimpleAgent) still show
"True board / Bot's view" as before.

## Records

Every game is saved, including one you abandon by starting another (marked "abandoned"). The page's **Past
games** list opens the replay of any saved game. `--rebuild-replay FILE...` rewrites replays from records, for
example after a viewer update or for an old `human_game_v1` record. To see what the agent was thinking in a
game, run `tools/analyze_human_game.py <record> --turns 9-11`. Notes from the games so far are in
`docs/human-games.md`.


Every finished game is saved to `--record-dir` (by default `runs/human_games/`, never committed) in two files:
- **JSON record** (`human_game_v2`):
  - the map, the episode seed and the settings;
  - the checkpoint, including its SHA-256;
  - the seats and the player alias;
  - every action of both sides, in order;
  - the result.
- **Replay HTML file**, next to the record.

The engine is deterministic given (map, seed, actions), so a record replays exactly. This command checks every
action and the outcome:

```bash
.venv/bin/python tools/play_vs_agent.py --verify-record runs/human_games/<file>.json
```

The human's moves are stored in a form the agent can learn from:
- in seat 0, as global action ids (the agent's own action space);
- in seat 1, as raw engine actions.

Use an alias with `--player`, not a real name.

## Towards a human protocol

Before reporting "the agent beats humans":
- **Maps:** fix a set of maps in advance.
  - The held-out `eval_test` maps need `--pool eval_test --confirm-test`. Under CLAUDE.md rule 1, use them
    only when a human has asked for that in the session.
  - Never tune on maps a human has played.
- **Seats:** play both seats.
- **Players:** record each player's level (for example a Polytopia multiplayer rating) with the alias.
- **Reporting:** report wins with a confidence interval, the checkpoint's SHA-256, and argmax or sampled.
