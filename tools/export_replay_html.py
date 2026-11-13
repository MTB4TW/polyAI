#!/usr/bin/env python
"""Generate an interactive HTML visual replay of a Polytopia game played by an AI model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure local Java is in PATH and JAVA_HOME
jdks = sorted((REPO_ROOT / ".runtime").glob("jdk-*/Contents/Home"))
if jdks:
    os.environ["JAVA_HOME"] = str(jdks[-1])
    os.environ["PATH"] = str(jdks[-1] / "bin") + os.pathsep + os.environ.get("PATH", "")

from pol_env.Tribes.py.register_env import TribesGymWrapper
from pol_env.Tribes.py.environment_contract import read_checkpoint_metadata
from py_rl.cleanrl.cleanrl.ppo import Agent
from tools.phase1_eval_core import PPOPolicy, load_verified_pool, legal_tensors


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Polytopia AI Interactive Visual Replay</title>
<style>
  :root {
    --bg: #1a1b26;
    --card: #24283b;
    --text: #c0caf5;
    --accent: #7aa2f7;
    --border: #414868;
    --success: #9ece6a;
    --warning: #e0af68;
    --danger: #f7768e;
  }
  body {
    margin: 0;
    padding: 20px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  h1 { margin: 0 0 10px 0; color: #fff; font-size: 24px; }
  .subtitle { color: #7982a9; margin-bottom: 20px; font-size: 14px; }
  .container {
    display: flex;
    gap: 24px;
    max-width: 1200px;
    width: 100%;
  }
  .left-pane {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: var(--card);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
  }
  #board-canvas {
    border-radius: 8px;
    background: #0f1017;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.6);
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 16px;
    width: 100%;
    justify-content: center;
  }
  button {
    background: var(--accent);
    color: #1a1b26;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
    cursor: pointer;
    font-size: 14px;
    transition: 0.15s;
  }
  button:hover { filter: brightness(1.15); transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  .slider-container {
    width: 100%;
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  input[type=range] {
    flex-grow: 1;
    accent-color: var(--accent);
  }
  .right-pane {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .stats-card {
    background: var(--card);
    padding: 16px;
    border-radius: 12px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  }
  .stat-box {
    text-align: center;
    background: #1f2335;
    padding: 12px;
    border-radius: 8px;
  }
  .stat-val { font-size: 24px; font-weight: bold; color: #fff; margin-top: 4px; }
  .stat-label { font-size: 11px; text-transform: uppercase; color: #7982a9; letter-spacing: 0.5px; }
  .feed-card {
    background: var(--card);
    padding: 16px;
    border-radius: 12px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    height: 380px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  }
  .feed-card h3 { margin: 0 0 10px 0; font-size: 16px; color: #fff; }
  .feed-list {
    overflow-y: auto;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-right: 6px;
  }
  .feed-item {
    background: #1f2335;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-left: 3px solid #565f89;
    cursor: pointer;
  }
  .feed-item.active {
    border-left-color: var(--accent);
    background: #292e42;
  }
  .feed-turn { font-weight: bold; color: var(--accent); margin-right: 8px; }
  .feed-action { color: #cfc9c2; flex-grow: 1; }
  .feed-delta { font-weight: bold; }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 14px;
    font-size: 12px;
    color: #7982a9;
    justify-content: center;
  }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-box { width: 14px; height: 14px; border-radius: 3px; }
</style>
</head>
<body>

<h1>The Battle of Polytopia — AI Match Replay</h1>
<div class="subtitle" id="match-meta">Loading replay data...</div>

<div class="container">
  <div class="left-pane">
    <canvas id="board-canvas" width="484" height="484"></canvas>
    <div class="slider-container">
      <span id="step-label">Step 0/0</span>
      <input type="range" id="step-slider" min="0" max="0" value="0">
    </div>
    <div class="controls">
      <button id="btn-prev">&lt; Prev</button>
      <button id="btn-play">Play &#9654;</button>
      <button id="btn-next">Next &gt;</button>
      <button id="btn-reset">&#8635; Start</button>
    </div>
    <div class="legend">
      <div class="legend-item"><div class="legend-box" style="background:#4a8505"></div> Plain</div>
      <div class="legend-item"><div class="legend-box" style="background:#265c19"></div> Forest</div>
      <div class="legend-item"><div class="legend-box" style="background:#707070"></div> Mountain</div>
      <div class="legend-item"><div class="legend-box" style="background:#1b6ca8"></div> Water</div>
      <div class="legend-item"><div class="legend-box" style="background:#c29c2d"></div> Village</div>
      <div class="legend-item"><div class="legend-box" style="background:#3b82f6; border:2px solid #fff;"></div> City</div>
      <div class="legend-item"><div class="legend-box" style="background:#ef4444; border-radius:50%;"></div> Unit</div>
    </div>
  </div>

  <div class="right-pane">
    <div class="stats-card">
      <div class="stat-box">
        <div class="stat-label">Turn</div>
        <div class="stat-val" id="stat-turn">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Stars Per Turn</div>
        <div class="stat-val" id="stat-spt" style="color:var(--success)">2.0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Stars Available</div>
        <div class="stat-val" id="stat-stars" style="color:var(--warning)">5</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Cities / Units</div>
        <div class="stat-val" id="stat-cities">1 / 1</div>
      </div>
    </div>

    <div class="feed-card">
      <h3>Match Action Log</h3>
      <div class="feed-list" id="action-feed"></div>
    </div>
  </div>
</div>

<script>
const REPLAY_DATA = REPLACE_WITH_REPLAY_DATA;

const canvas = document.getElementById("board-canvas");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("step-slider");
const stepLabel = document.getElementById("step-label");
const metaLabel = document.getElementById("match-meta");
const feedList = document.getElementById("action-feed");

const statTurn = document.getElementById("stat-turn");
const statSpt = document.getElementById("stat-spt");
const statStars = document.getElementById("stat-stars");
const statCities = document.getElementById("stat-cities");

const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnPlay = document.getElementById("btn-play");
const btnReset = document.getElementById("btn-reset");

let currentStep = 0;
let isPlaying = false;
let playInterval = null;

const TERRAIN_COLORS = {
  0: "#4a8505", // PLAIN
  1: "#2980b9", // SHALLOW_WATER
  2: "#1b4f72", // DEEP_WATER
  3: "#707b7c", // MOUNTAIN
  4: "#d4ac0d", // VILLAGE
  5: "#2563eb", // CITY
  6: "#1e824c"  // FOREST
};

function init() {
  metaLabel.textContent = `Map: ${REPLAY_DATA.map_name} | Policy: ${REPLAY_DATA.policy_name} | Final Turn-10 SPT: ${REPLAY_DATA.final_spt.toFixed(1)}`;
  slider.max = REPLAY_DATA.steps.length - 1;
  slider.value = 0;
  
  // Build feed
  feedList.innerHTML = "";
  REPLAY_DATA.steps.forEach((st, idx) => {
    const item = document.createElement("div");
    item.className = "feed-item" + (idx === 0 ? " active" : "");
    item.id = `feed-item-${idx}`;
    item.onclick = () => setStep(idx);
    item.innerHTML = `
      <span class="feed-turn">T${st.turn}</span>
      <span class="feed-action">${st.action_repr}</span>
      <span class="feed-delta" style="color:${st.spt_delta > 0 ? 'var(--success)' : '#7982a9'}">${st.spt_delta > 0 ? '+' + st.spt_delta.toFixed(1) + ' SPT' : ''}</span>
    `;
    feedList.appendChild(item);
  });
  
  renderStep(0);
}

function renderStep(idx) {
  currentStep = idx;
  slider.value = idx;
  stepLabel.textContent = `Step ${idx}/${REPLAY_DATA.steps.length - 1}`;
  
  const st = REPLAY_DATA.steps[idx];
  statTurn.textContent = st.turn;
  statSpt.textContent = st.spt.toFixed(1);
  statStars.textContent = st.stars;
  statCities.textContent = `${st.city_count} / ${st.unit_count}`;
  
  // Highlight feed
  document.querySelectorAll(".feed-item").forEach(el => el.classList.remove("active"));
  const activeFeed = document.getElementById(`feed-item-${idx}`);
  if (activeFeed) {
    activeFeed.classList.add("active");
    activeFeed.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Draw board
  const board = st.board;
  const size = board.terrain.length;
  const tileSize = canvas.width / size;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const terr = board.terrain[r][c];
      const fog = board.fog ? board.fog[r][c] : 0;
      
      const x = c * tileSize;
      const y = r * tileSize;
      
      // Base tile
      ctx.fillStyle = TERRAIN_COLORS[terr] || "#333";
      ctx.fillRect(x, y, tileSize, tileSize);
      
      // Grid border
      ctx.strokeStyle = "rgba(0,0,0,0.15)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, tileSize, tileSize);
      
      // Building or Forest overlay
      if (terr === 6) { // Forest
        ctx.fillStyle = "#145a32";
        ctx.beginPath();
        ctx.arc(x + tileSize * 0.5, y + tileSize * 0.5, tileSize * 0.28, 0, Math.PI * 2);
        ctx.fill();
      } else if (terr === 4) { // Village
        ctx.fillStyle = "#f1c40f";
        ctx.beginPath();
        ctx.arc(x + tileSize * 0.5, y + tileSize * 0.5, tileSize * 0.22, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#7d6608";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("V", x + tileSize * 0.5, y + tileSize * 0.5 + 3);
      }
      
      // Cities
      const cid = board.city_id ? board.city_id[r][c] : -1;
      if (cid !== -1) {
        ctx.fillStyle = "#3b82f6";
        ctx.fillRect(x + 4, y + 4, tileSize - 8, tileSize - 8);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 4, y + 4, tileSize - 8, tileSize - 8);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("★", x + tileSize * 0.5, y + tileSize * 0.5 + 4);
      }
      
      // Units
      const uid = board.unit_id ? board.unit_id[r][c] : 0;
      if (uid > 0) {
        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(x + tileSize * 0.5, y + tileSize * 0.5, tileSize * 0.32, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("U", x + tileSize * 0.5, y + tileSize * 0.5 + 3);
      }
      
      // Fog of War
      if (fog) {
        ctx.fillStyle = "rgba(15, 16, 23, 0.88)";
        ctx.fillRect(x, y, tileSize, tileSize);
      }
    }
  }
}

function setStep(idx) {
  if (idx < 0) idx = 0;
  if (idx >= REPLAY_DATA.steps.length) idx = REPLAY_DATA.steps.length - 1;
  renderStep(idx);
}

slider.oninput = (e) => setStep(parseInt(e.target.value));
btnPrev.onclick = () => setStep(currentStep - 1);
btnNext.onclick = () => setStep(currentStep + 1);
btnReset.onclick = () => setStep(0);

btnPlay.onclick = () => {
  if (isPlaying) {
    clearInterval(playInterval);
    isPlaying = false;
    btnPlay.textContent = "Play ▶";
  } else {
    isPlaying = true;
    btnPlay.textContent = "Pause ❚❚";
    playInterval = setInterval(() => {
      if (currentStep < REPLAY_DATA.steps.length - 1) {
        setStep(currentStep + 1);
      } else {
        clearInterval(playInterval);
        isPlaying = false;
        btnPlay.textContent = "Play ▶";
      }
    }, 600);
  }
};

window.onload = init;
</script>
</body>
</html>
"""


def extract_board_state(env: TribesGymWrapper) -> dict:
    raw_obs = getattr(env.tribes_env, "_last_obs", {})
    board = raw_obs.get("board", {}) if isinstance(raw_obs, dict) else {}
    terrain = board.get("terrain", [])
    unit_id = board.get("unitID", [])
    city_id = board.get("cityID", [])
    fog = board.get("fog", [])
    return {
        "terrain": terrain,
        "unit_id": unit_id,
        "city_id": city_id,
        "fog": fog,
    }


def main():
    parser = argparse.ArgumentParser(description="Export interactive Polytopia AI visual match replay")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to .cleanrl_model checkpoint")
    parser.add_argument("--map-path", type=Path, default=Path("map_002814.csv"), help="Path or filename of map to replay")
    parser.add_argument("--output-html", type=Path, default=Path("replay.html"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model_path
    if model_path is None or not model_path.exists():
        if model_path is not None and model_path.is_dir():
            candidates = sorted(model_path.glob("**/*.cleanrl_model"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                model_path = candidates[0]
        if model_path is None or not model_path.exists():
            candidates = sorted((REPO_ROOT / "runs").glob("**/*.cleanrl_model"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                model_path = candidates[0]
            elif model_path is not None and not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            else:
                raise FileNotFoundError("No .cleanrl_model found under runs/")

    print(f"Using checkpoint: {model_path}")

    # Resolve map path if given as just a filename
    resolved_map = args.map_path
    if not resolved_map.exists():
        candidates = list((REPO_ROOT / "pol_env/Tribes/levels/phase1_pool_bardur_real").rglob(args.map_path.name))
        if candidates:
            resolved_map = candidates[0]
        else:
            raise FileNotFoundError(f"Map file not found: {args.map_path}")

    device = torch.device(args.device)

    # 1. Environment initialization
    os.environ["POLYVISION_SOLO_NO_OPPONENT_MODE"] = "1"
    os.environ["POLYVISION_INFO_MODE"] = "fast"
    env = TribesGymWrapper()
    env._level_pool = [str(resolved_map.resolve())]
    env._level_pool_size = 1

    checkpoint = torch.load(model_path, map_location=device)

    # 2. Model initialization
    from types import SimpleNamespace
    adapter = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)

    meta = None
    try:
        meta = read_checkpoint_metadata(str(model_path))
    except Exception:
        pass

    # Infer arch from weights if not in metadata
    has_conv = any("spatial_encoder" in k or "res_blocks" in k for k in checkpoint.keys())
    default_arch = "spatial_resnet" if has_conv else "mlp"

    arch = meta.get("network_arch", default_arch) if meta else default_arch
    actor_mode = meta.get("actor_mode", "legal_features") if meta else "legal_features"
    max_legal = meta.get("max_legal_actions", 256) if meta else 256
    feat_dim = meta.get("legal_action_feature_dim", 47) if meta else 47

    from py_rl.cleanrl.cleanrl.agent_factory import make_agent
    from py_rl.cleanrl.cleanrl.obs_normalization import observation_normalization_from_metadata
    agent = make_agent(adapter, network_arch=arch, actor_mode=actor_mode,
                       max_legal_actions=max_legal, legal_action_feature_dim=feat_dim,
                       observation_normalization=observation_normalization_from_metadata(meta))
    agent.load_state_dict(checkpoint)
    agent.to(device)
    agent.eval()

    policy = PPOPolicy(agent, device=device, sampled=False, policy_seed=42)

    # 3. Step through episode and record state
    obs, info = env.reset(seed=1)
    steps_data = []

    last_spt = float(info.get("spt", 2.0))
    initial_board = extract_board_state(env)
    steps_data.append({
        "step": 0,
        "turn": int(info.get("turn_count", 0)),
        "spt": last_spt,
        "spt_delta": 0.0,
        "stars": int(info.get("stars", 5)),
        "city_count": int(info.get("city_count", 1)),
        "unit_count": int(info.get("unit_count", 1)),
        "action_repr": "Game Start / Initial State",
        "board": initial_board,
    })

    terminated = truncated = False
    step_count = 0

    print(f"Replaying game on {args.map_path.name}...")

    while not (terminated or truncated) and step_count < 200:
        action, slot = policy.choose_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        curr_spt = float(info.get("spt", last_spt))
        spt_delta = curr_spt - last_spt
        last_spt = curr_spt

        action_type = str(info.get("selected_action_type", "ACTION"))
        board_snapshot = extract_board_state(env)

        steps_data.append({
            "step": step_count,
            "turn": int(info.get("turn_count", 0)),
            "spt": curr_spt,
            "spt_delta": float(spt_delta),
            "stars": int(info.get("stars", 0)),
            "city_count": int(info.get("city_count", 1)),
            "unit_count": int(info.get("unit_count", 1)),
            "action_repr": f"{action_type} (Reward: {float(reward):+.2f})",
            "board": board_snapshot,
        })

    env.close()

    replay_payload = {
        "map_name": resolved_map.name,
        "policy_name": f"PPO ({arch})",
        "final_spt": float(last_spt),
        "steps": steps_data,
    }

    # 4. Write HTML output
    html_content = HTML_TEMPLATE.replace("REPLACE_WITH_REPLAY_DATA", json.dumps(replay_payload))
    args.output_html.write_text(html_content, encoding="utf-8")
    print(f"Successfully generated interactive visual replay: {args.output_html.resolve()}")
    print(f"Open {args.output_html} in Chrome or Safari to watch the replay!")


if __name__ == "__main__":
    main()
