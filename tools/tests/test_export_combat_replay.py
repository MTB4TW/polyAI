"""Event reconstruction of tools/export_combat_replay.py (no Java needed)."""

import unittest

from tools.export_combat_replay import describe_action, diff_events


def board(units, cities):
    return {"terrain": [], "units": units, "cities": cities, "tribes": {}, "tick": 0}


def unit(x, y, owner, hp=10, t=0):
    return {"x": x, "y": y, "t": t, "o": owner, "hp": hp, "vet": False, "k": 0}


def city(x, y, owner, cap=False):
    return {"x": x, "y": y, "o": owner, "cap": cap, "lvl": 1, "walls": False}


class CombatReplayEventsTest(unittest.TestCase):
    def test_diff_events_moves_damage_kills_and_captures(self):
        before = board({"2": unit(1, 1, 0), "4": unit(3, 3, 1, hp=10), "5": unit(5, 5, 1)}, {"1": city(0, 0, 0), "3": city(6, 6, 1, cap=True)})
        after = board({"2": unit(2, 2, 0, hp=6), "4": unit(3, 3, 1, hp=4), "7": unit(0, 0, 0, t=1)},
                      {"1": city(0, 0, 0), "3": city(6, 6, 0, cap=True), "8": city(4, 4, 0)})
        kinds = {(e["kind"], e["side"]) for e in diff_events(before, after)}
        assert kinds == {("move", 0), ("damaged", 0), ("damaged", 1), ("killed", 1), ("train", 0), ("capture", 0)}
        texts = [e["text"] for e in diff_events(before, after)]
        assert "CAPITAL at (6,6) captured" in texts
        assert "Village at (4,4) captured (new city)" in texts


    def test_describe_action_uses_board_changes(self):
        before = board({"6": unit(7, 3, 0)}, {})
        after = board({"6": unit(7, 3, 0, hp=2)}, {})
        events = diff_events(before, after)
        text = describe_action({"type": "ATTACK", "repr": "ATTACK by unit 6 to unit 9"}, before, events)
        assert text.startswith("Warrior (7,3) attacks") and "HP 10 → 2" in text
        moved = diff_events(before, board({"6": unit(6, 2, 0)}, {}))
        assert describe_action({"type": "MOVE", "repr": "MOVE by unit 6 to 6 : 2"}, before, moved) == "Move Warrior (7,3) → (6,2)"
        assert describe_action({"type": "END_TURN"}, before, []) == "End turn"


if __name__ == "__main__":
    unittest.main()
