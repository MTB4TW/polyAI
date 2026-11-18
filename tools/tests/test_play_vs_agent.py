"""Tests for tools/play_vs_agent.py: pure helpers, plus engine-backed games when Java is available."""

import json
import os
import random
import shutil
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from tools import play_vs_agent as P

HAVE_JAVA = shutil.which("java") is not None or bool(os.environ.get("JAVA_HOME"))


class HelperTests(unittest.TestCase):
    def test_legal_view_hides_diplomacy_and_keeps_raw_indices(self):
        actions = [
            {"idx": 0, "type": "RESEARCH_TECH", "tech_type": "CLIMBING", "star_cost": 5},
            {"idx": 1, "type": "SEND_STARS", "star_cost": 0},
            {"idx": 2, "type": "END_TURN"},
            {"idx": 3, "type": "MOVE", "src_x": 7, "src_y": 8, "target_x": 6, "target_y": 8},
        ]
        view = P.legal_view(actions)
        self.assertEqual([a["i"] for a in view], [0, 2, 3])
        self.assertEqual(view[0]["text"], "Research Climbing")
        self.assertEqual(view[0]["cost"], 5)
        self.assertEqual((view[2]["src"], view[2]["dst"]), ([7, 8], [6, 8]))
        with self.assertRaises(RuntimeError):
            P.legal_view([{"idx": 5, "type": "END_TURN"}])

    def test_seen_changes_does_not_call_units_leaving_sight_dead(self):
        fog, plain = P.FOG, 0
        terrain = [[plain, fog], [plain, plain]]
        unit = lambda uid, o, x, y, hp=10, t=0: {"id": uid, "o": o, "x": x, "y": y, "hp": hp, "t": t}
        before = {"terrain": terrain, "cities": [], "units": [unit(1, 0, 0, 1), unit(2, 0, 1, 1), unit(3, 1, 0, 0)]}
        after = {"terrain": terrain, "cities": [], "units": [unit(3, 1, 0, 0, hp=6), unit(4, 0, 1, 0)]}
        lines = P.seen_changes(before, after)
        self.assertNotIn("Enemy Warrior at (0,1) was killed", lines)        # walked into fog
        self.assertIn("Enemy Warrior at (1,1) was killed", lines)            # visible tile
        self.assertIn("Your Warrior at (0,0) took 4 damage (HP 6)", lines)
        self.assertIn("Enemy Warrior seen at (1,0)", lines)
        self.assertEqual(P.seen_changes(None, after), [])


@unittest.skipUnless(HAVE_JAVA, "needs the Java engine")
class EngineGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from types import SimpleNamespace
        cls.tmp = Path(tempfile.mkdtemp())
        args = SimpleNamespace(model_path=None, pool="eval_validation", rules="tribes_2020", max_turns=4)
        cls.env, cls.settings = P.make_env(args)
        cls.agent = P.AgentPlayer(None, cls.env, False, 0)
        cls.maps = P.load_maps("eval_validation")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_scripted_game_is_recorded_and_replays_to_the_same_outcome(self):
        rng = random.Random(3)

        def human(state):  # random legal moves, ending the turn after a few
            legal = state["legal"]
            ends = [a["i"] for a in legal if a["type"] == "END_TURN"]
            others = [a["i"] for a in legal if a["type"] != "END_TURN"]
            return rng.choice(others) if others and rng.random() < 0.8 else ends[0]

        game_map = self.maps[0]
        session = P.GameSession(self.env, self.agent, str(game_map.path), 1234, self.settings, self.tmp, "tester",
                                human_policy=human)
        session._run()
        self.assertIsNone(session.error, session.error)
        state = session.snapshot()
        self.assertEqual(state["phase"], "over")
        self.assertIn(state["agent_outcome"], ("win", "loss", "timeout"))
        self.assertTrue(any(a["side"] == P.HUMAN_TRIBE for a in session.history))
        self.assertTrue(state["agent_moves"], "the agent's moves are revealed at the end")
        self.assertTrue(all(e["side"] in ("you", "agent") for e in state["log"]), "no hidden agent moves in the live log")
        record = json.loads(Path(state["record"]).read_text())
        self.assertEqual(record["record_version"], P.RECORD_VERSION)
        self.assertEqual(P.replay_record(self.env, record), record["agent_outcome"])

    def test_browser_flow_over_http(self):
        from http.server import ThreadingHTTPServer

        server = P.Server(self.env, self.agent, self.maps, self.settings, self.tmp, "", seed=5)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def get():
            with urllib.request.urlopen(base + "/api/state", timeout=30) as r:
                return json.loads(r.read())

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def wait_for(pred, timeout=120):
            t0 = time.time()
            while time.time() - t0 < timeout:
                s = get()
                if pred(s):
                    return s
                time.sleep(0.05)
            self.fail("timed out waiting for the game")

        try:
            with urllib.request.urlopen(base + "/", timeout=30) as r:
                self.assertIn(b"Play the Agent", r.read())
            self.assertEqual(post("/api/new", {"map": self.maps[1].filename, "seed": 77})[0], 200)
            s = wait_for(lambda s: s.get("phase") in ("your_turn", "over"))
            self.assertEqual(s["map"], self.maps[1].filename)
            if s["phase"] == "your_turn":
                self.assertTrue(all(u["o"] in (0, 1) for u in s["board"]["units"]))
                self.assertEqual(post("/api/act", {"i": s["legal"][0]["i"], "version": s["version"] - 1})[0], 409)  # stale
                end = [a for a in s["legal"] if a["type"] == "END_TURN"][0]
                self.assertEqual(post("/api/act", {"i": end["i"], "version": s["version"]})[0], 200)
                s2 = wait_for(lambda x: x["version"] > s["version"] and x["phase"] in ("your_turn", "over"))
                if s2["phase"] == "your_turn":
                    self.assertEqual(s2["turn"], s["turn"] + 1)
                    self.assertEqual(post("/api/resign", {})[0], 200)
                    s3 = wait_for(lambda x: x["phase"] == "over")
                    self.assertEqual(s3["result"], "loss")
                    record = json.loads(Path(s3["record"]).read_text())
                    self.assertEqual(P.replay_record(self.env, record), "human_resigned")
        finally:
            httpd.shutdown()
            httpd.server_close()
            if server.session is not None:
                server.session.abort()


if __name__ == "__main__":
    unittest.main()
