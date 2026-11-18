"""Tests for tools/play_vs_agent.py: pure helpers, plus engine-backed games when Java is available."""

import json
import os
import random
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
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
        view = P.legal_view(P.raw_pairs(actions))
        self.assertEqual([a["i"] for a in view], [0, 2, 3])
        self.assertEqual(view[0]["text"], "Research Climbing")
        self.assertEqual(view[0]["cost"], 5)
        self.assertEqual((view[2]["src"], view[2]["dst"]), ([7, 8], [6, 8]))
        with self.assertRaises(RuntimeError):
            P.legal_view(P.raw_pairs([{"idx": 5, "type": "END_TURN"}]))
        self.assertEqual([a["i"] for a in P.legal_view([(9001, actions[2])])], [9001])  # wrapper ids pass through

    def test_seen_changes_does_not_call_units_leaving_sight_dead(self):
        fog, plain = P.FOG, 0
        terrain = [[plain, fog], [plain, plain]]
        unit = lambda uid, o, x, y, hp=10, t=0: {"id": uid, "o": o, "x": x, "y": y, "hp": hp, "t": t}
        before = {"terrain": terrain, "cities": [], "units": [unit(1, 0, 0, 1), unit(2, 0, 1, 1), unit(3, 1, 0, 0)]}
        after = {"terrain": terrain, "cities": [], "units": [unit(3, 1, 0, 0, hp=6), unit(4, 0, 1, 0)]}
        lines = P.seen_changes(before, after, me=1)
        self.assertNotIn("Enemy Warrior at (0,1) was killed", lines)        # walked into fog
        self.assertIn("Enemy Warrior at (1,1) was killed", lines)            # visible tile
        self.assertIn("Your Warrior at (0,0) took 4 damage (HP 6)", lines)
        self.assertIn("Enemy Warrior seen at (1,0)", lines)
        self.assertEqual(P.seen_changes(None, after, me=1), [])
        self.assertIn("Your Warrior at (1,1) was killed", P.seen_changes(before, after, me=0))

    def test_view_of_shows_own_pieces_and_everything_on_explored_tiles(self):
        full = {"terrain": [[0, 0], [0, 0]],
                "units": {"1": {"x": 0, "y": 0, "o": 0}, "2": {"x": 1, "y": 1, "o": 1}, "3": {"x": 0, "y": 1, "o": 1}},
                "cities": {"5": {"x": 1, "y": 1, "o": 1}, "6": {"x": 1, "y": 0, "o": 0}}}
        mask = [[1, 1], [0, 0]]
        v = P.view_of(full, mask, 0)
        self.assertEqual(v["units"], ["1", "3"])
        self.assertEqual(v["cities"], ["6"])
        self.assertEqual(P.view_of(full, None, 1)["units"], ["2", "3"])

    def test_find_models_reads_combat_checkpoints_only(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "Tribes-v0__my_league_run__3__1794900049"
            run.mkdir()
            version = "combat_v5_scaled_scalars_horizon_20+rules_polytopia_2_15_r2"
            for name, ver in (("model_checkpoint_400000.cleanrl_model", version), ("my_league_run.cleanrl_model", version),
                              ("phase1.cleanrl_model", "phase1_v5"), ("orphan.cleanrl_model", None)):
                (run / name).write_bytes(b"weights")
                if ver:
                    (run / (name + ".action_interface.json")).write_text(json.dumps(
                        {"phase1_environment_version": ver, "network_arch": "spatial_resnet_composed"}))
            models = P.find_models([str(run / "*.cleanrl_model")])
            self.assertEqual(sorted(m.path.name for m in models),
                             ["model_checkpoint_400000.cleanrl_model", "my_league_run.cleanrl_model"])
            step = [m for m in models if m.step][0]
            self.assertEqual((step.run, step.seed, step.step, step.max_turns, step.rules),
                             ("my_league_run", 3, 400000, 20, "polytopia_2_15"))
            self.assertIn("step 400,000", step.label())
            self.assertEqual(len(step.public(with_sha=True)["sha256"]), 64)


@unittest.skipUnless(HAVE_JAVA, "needs the Java engine")
class EngineGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env, cls.settings = P.make_env(None, "eval_validation", "polytopia_2_15", 4)
        cls.agent = P.AgentPlayer(None, cls.env, False, 0)
        cls.maps = P.load_maps("eval_validation")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def play(self, human_tribe, seed):
        rng = random.Random(seed)
        self.seen_states = []

        def human(state):  # random legal moves, ending the turn after a few
            self.seen_states.append(state)
            legal = state["legal"]
            ends = [a["i"] for a in legal if a["type"] == "END_TURN"]
            others = [a["i"] for a in legal if a["type"] != "END_TURN"]
            return rng.choice(others) if others and rng.random() < 0.8 else ends[0]

        session = P.GameSession(self.env, self.agent, self.maps[seed % len(self.maps)], 1000 + seed, self.settings,
                                self.tmp, "tester", human_tribe=human_tribe, human_policy=human)
        session._run()
        self.assertIsNone(session.error, session.error)
        return session

    def check_game(self, session):
        state = session.snapshot()
        self.assertEqual(state["phase"], "over")
        self.assertIn(state["tribe0_outcome"], ("win", "loss", "timeout"))
        sides = {a["side"] for a in session.history}
        self.assertEqual(sides, {0, 1})
        # replay: true board plus both players' views on every frame; fog hides the other side's pieces
        payload = session.replay_payload()
        self.assertEqual(payload["mark_pov"], session.human)
        frames = payload["frames"]
        self.assertTrue(payload["lighthouses"])
        self.assertTrue(all("resource" in f["full"] and all("mh" in u for u in f["full"]["units"].values()) for f in frames))
        attacks = [f for f in frames if f["action"] == "ATTACK"]
        self.assertTrue(all(f.get("src") and f.get("dst") for f in attacks))
        self.assertGreater(len(frames), 5)
        self.assertEqual(frames[0]["label"], "Start")
        self.assertTrue(all(set(f["views"]) == {"0", "1"} for f in frames))
        self.assertTrue({f["actor"] for f in frames[1:]} >= {0, 1})
        f0 = frames[0]
        enemy_of_0 = [uid for uid, u in f0["full"]["units"].items() if u["o"] == 1]
        self.assertTrue(enemy_of_0 and not set(enemy_of_0) <= set(f0["views"]["0"]["units"]), "fog hides the enemy at the start")
        self.assertTrue(all(any(sum(f["views"][t]["visible"], [])) for f in frames for t in ("0", "1")),
                        "both views known on every frame (back-filled before tribe 1's first turn)")
        self.assertIn("/*REPLAY_DATA*/", (P.REPO_ROOT / "tools/combat_replay_template.html").read_text())
        self.assertNotIn("/*REPLAY_DATA*/null", session.replay_html())
        self.assertTrue(Path(state["replay"]).is_file())
        # the record replays to the same actions and outcome
        record = json.loads(Path(state["record"]).read_text())
        self.assertEqual(record["record_version"], P.RECORD_VERSION)
        outcome, history, _ = P.replay_record(self.env, record, session.map_info)
        self.assertEqual(outcome, record["tribe0_outcome"])
        strip = lambda h: [(a["side"], a.get("gid"), a.get("raw"), a["repr"]) for a in h]
        self.assertEqual(strip(history), strip(record["actions"]))

    def check_live_view(self, session):
        """What the browser gets on the human's turns: exact max HP, lighthouses, and the agent's visible moves."""
        states = self.seen_states
        board = states[0]["board"]
        self.assertTrue(all(u["maxhp"] >= u["hp"] and u["maxhp"] >= 10 for u in board["units"] if u["t"] in (0, 1)))
        self.assertEqual(len(board["lighthouses"]), 4)
        steps = [st["agent_steps"] for st in states if st.get("agent_steps")]
        self.assertTrue(steps, "agent moves are handed to the browser")
        for batch in steps:
            for step in batch["steps"]:
                self.assertTrue(step["text"])
                b = step["board"]
                # the step board is the human's fogged view: no agent unit on a fogged tile
                for u in b["units"]:
                    if u["o"] != session.human:
                        self.assertNotEqual(b["terrain"][u["x"]][u["y"]], P.FOG)

    def test_human_second_seat(self):
        session = self.play(1, 3)
        self.check_game(session)
        self.check_live_view(session)

    def test_human_first_seat_agent_plays_tribe_one(self):
        session = self.play(0, 4)
        self.check_game(session)
        self.check_live_view(session)
        human_moves = [a for a in session.history if a["side"] == 0]
        self.assertTrue(all("gid" in a and "raw" not in a for a in human_moves), "seat 0 acts through the wrapper")
        agent_moves = [a for a in session.history if a["side"] == 1]
        self.assertTrue(agent_moves and all("gid" in a and "raw" in a for a in agent_moves))

    def test_abandoned_game_is_saved_listed_and_rebuilt(self):
        moves = iter(range(3))

        def human(state):  # two moves, then the player starts another game
            if next(moves) == 2:
                raise P.GameAborted()
            others = [a["i"] for a in state["legal"] if a["type"] != "END_TURN"]
            return others[0] if others else state["legal"][0]["i"]

        d = self.tmp / "abandon"
        session = P.GameSession(self.env, self.agent, self.maps[2], 55, self.settings, d, "", human_tribe=1, human_policy=human)
        session._run()
        self.assertEqual(session.outcome, "aborted")
        record = json.loads(Path(session.state["record"]).read_text())
        self.assertEqual(record["tribe0_outcome"], "abandoned")
        games = P.list_games(d)
        self.assertEqual((len(games), games[0]["result"]), (1, "abandoned"))
        self.assertTrue(games[0]["replay"])
        html = Path(session.state["replay"])
        html.unlink()
        self.assertTrue(P.verify(Path(session.state["record"]), "eval_validation", rebuild=True))
        self.assertTrue(html.is_file())

    def test_browser_flow_over_http(self):
        from http.server import ThreadingHTTPServer

        server = P.Server([], self.maps, "eval_validation", self.tmp, "", seed=5, rules="tribes_2020", max_turns=4)
        server.env, server.env_settings = self.env, dict(self.settings, _wanted=("tribes_2020", 4, 512, None))
        server.rules = "tribes_2020"  # matches _wanted, so the shared env is reused
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def get(path="/api/state"):
            with urllib.request.urlopen(base + path, timeout=30) as r:
                return json.loads(r.read())

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                with e:
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
            catalog = get("/api/catalog")
            self.assertIn(self.maps[1].filename, catalog["maps"])
            self.assertEqual(post("/api/new", {"model": "random", "map": self.maps[1].filename, "seed": 77, "seat": "first"})[0], 200)
            s = wait_for(lambda s: s.get("phase") in ("your_turn", "over"))
            self.assertEqual((s["map"], s["human_tribe"], s["you_move"], s["model"]), (self.maps[1].filename, 0, "first", None))
            try:
                urllib.request.urlopen(base + "/replay", timeout=30)
                self.fail("the replay must not be served during the game")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 409)
                e.close()
            self.assertEqual(s["phase"], "your_turn")
            self.assertEqual(post("/api/act", {"i": s["legal"][0]["i"], "version": s["version"] - 1})[0], 409)  # stale
            end = [a for a in s["legal"] if a["type"] == "END_TURN"][0]
            self.assertEqual(post("/api/act", {"i": end["i"], "version": s["version"]})[0], 200)
            s2 = wait_for(lambda x: x["version"] > s["version"] and x["phase"] in ("your_turn", "over"))
            self.assertEqual(s2["phase"], "your_turn")
            self.assertEqual(s2["turn"], s["turn"] + 1)
            self.assertEqual(post("/api/resign", {})[0], 200)
            s3 = wait_for(lambda x: x["phase"] == "over")
            self.assertEqual(s3["result"], "loss")
            with urllib.request.urlopen(base + "/replay", timeout=30) as r:
                html = r.read().decode()
            self.assertIn('"short_names"', html)
            games = get("/api/games")["games"]
            self.assertTrue(any(g["file"] == Path(s3["record"]).name for g in games))
            with urllib.request.urlopen(base + "/saved/" + Path(s3["replay"]).name, timeout=30) as r:
                self.assertEqual(r.status, 200)
                self.assertIn(b"const D = {", r.read())
            try:
                urllib.request.urlopen(base + "/saved/..%2F..%2Fsecret.html", timeout=30)
                self.fail("paths outside the record folder must not be served")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 404)
                e.close()
            record = json.loads(Path(s3["record"]).read_text())
            outcome, _, _ = P.replay_record(self.env, record, self.maps[1])
            self.assertEqual(outcome, "resigned")
        finally:
            httpd.shutdown()
            httpd.server_close()
            if server.session is not None:
                server.session.abort()


if __name__ == "__main__":
    unittest.main()
