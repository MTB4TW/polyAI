# Human games against the agent: notes

Games played with `tools/play_vs_agent.py`. Records and replays are in `runs/human_games/` on the machine
that served the game (never committed). Every record below replays move for move (`--verify-record`).

Each game is a diagnostic: one game, one map. None of it is a canonical result.

## 2026-11-17: Kevin vs `combat_pcritic_t1_league_swap_4m` (argmax), Kevin moving second

| Record | Map | Result | Notes |
|---|---|---|---|
| `20261118T035103Z_map_2p_030008_1633342428` | map_2p_030008 | agent won, turn 6 | Step 3.8M (`human_game_v1`). Played well: went for a Giant as early as possible. |
| `20261118T043258Z_map_2p_030108_1913469291_human1` | map_2p_030108 | agent won, turn 12 | Final model. |
| `20261118T045025Z_map_2p_030059_1464236660_human1` | map_2p_030059 | **Kevin won, turn 17** | Final model. The agent was winning and had a forced win, then lost. |

### What went wrong in the game the agent lost (map_2p_030059)

Kevin's observations:
- the agent had a forced win around turn 10 but did not take the capital;
- it left a village uncaptured;
- it shuffled units sideways instead of pushing;
- it did not recapture a city, because its rider moved off the city instead of staying.

`tools/analyze_human_game.py <record> --turns 10` replays the game with the checkpoint and prints the policy's
probabilities at every decision. It finds the following:

- **Turn 10, the capital: no focus fire.**
  - The position: Kevin's capital (7,4) was held by one Warrior on 10 HP. His only other units were two
    damaged Riders far away, at (1,0) with 4 HP and (9,9) with 6 HP.
  - The agent's forces: Warriors adjacent or able to reach, at (6,3) 5 HP, (6,4) 9 HP and (6,5) 6 HP, plus two
    Riders.
  - What it did: it attacked the defender with only two units. The 9 HP Warrior next to the capital chose
    RECOVER (p 0.42) over attacking (p 0.09). Two Warriors spent the turn on moves.
  - The defender survived, so no unit could step onto the capital to capture it next turn.
  - Killing the defender and moving in was almost certainly decisive, given how far away and weak Kevin's
    other units were.
- **No capture was ever legal.** The engine allows CAPTURE only for a unit on the city that has not moved or
  attacked this turn (`Capture.isFeasible`: the unit must be fresh). In practice, that is a unit that moved in
  on the previous turn. So the failure is in the setup (kill, then occupy), not in a refusal to press CAPTURE.
- **Turn 10, the village at (5,7): an argmax coin flip.**
  - The move onto the village scored 0.349, against 0.351 for the move next to it, so argmax picked the
    neighbour.
  - On turn 11 the village move scored 0.000.
  - Sampling, or any tie-breaking towards captures, would have taken it half the time.
- **Sideways moves.** No unit returned to the tile it held two turns earlier, so there was no strict
  back-and-forth.
  - From turn 10, though, a growing share of moves kept the same distance to the enemy capital: 2 of 6 on
    turn 10, 3 of 8 on turn 12.
  - This fits Kevin's "side by side" shuffling.
- **The rider leaving the city:** not located yet. Note the turn next time.

### Likely causes and what to try

- **SimpleAgent rarely defends a capital the way a human does.** A single fresh defender standing on the
  capital is a position the agent may seldom have had to break.
  - Measure this: count, in training or evaluation games, how often the capital is captured while it had a
    defender on it at the start of the turn.
- **Critical-position restarts** (roadmap Stage 2.4). The start of turn 10 in this game is exactly the kind of
  position to restart training from. The record replays to it.
- **Play-time search** (`ppo_search` in `tools/evaluate_combat.py`). The tactical search plans
  kill-then-occupy sequences. Check whether it finds the turn-10 line from this position. If it does, the human
  games should be played with policy + search, which is what the evaluator already offers.
- **League.** The 4M league run wins 71–87% against snapshots. A human who keeps a defender on the capital
  exposes a weakness that frozen copies of the agent may not.

## Ideas logged for later

- **Live win odds during a game.**
  - The checkpoint's critic is the privileged critic: it sees the whole board, so showing its estimate to the
    human would leak hidden information during play. It can be shown in the replay afterwards.
  - For a live display, train a fogged value head on stored games (the game store's `value` learner) and show
    that instead.
