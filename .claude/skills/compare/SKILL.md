---
name: compare
description: Compare a finished PolyAI evaluation with the current reference map by map, record it in docs/experiments.md, and prepare the results for commit. Use when an evaluation or experiment has finished, or when asked how a model compares. Arguments - candidate evaluation folder, optional baseline folder.
---

# Compare and record a result

1. If the results are still on Farmshare, fetch them from the Mac:
   `FARMSHARE_USER=<sunetid> tools/fetch_results.sh <name or evaluation id>`
2. Baseline defaults to the current reference named in `docs/experiments.md`
   (`20260827_phase1_v5_parity002_seed3_16m_terminal_spt_validation_canonical`).
3. Run, saving the comparison:
   `python3 tools/compare_evals.py <baseline> <candidate> --output outputs/comparisons/<candidate>_vs_reference.json`
   If it reports differing maps or seeds, stop and tell the user; the runs are not comparable.
4. Add a row to the top of the table in `docs/experiments.md`: date, name, arch, steps, seed, who ran it,
   validation SPT, Δ vs reference with its 95% CI, the evaluation folder, and a one-line note.
   Mention training-budget differences (e.g. 5M vs 16M steps) in the note.
5. Report the result in plain terms. A CI that includes 0 means no detectable difference; say so.
6. Stage the evaluation folder, `outputs/experiments/<name>.json`, the comparison JSON and
   `docs/experiments.md` on a branch, and offer to commit and open a pull request. Do not push without being asked.
