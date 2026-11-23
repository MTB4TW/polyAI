#!/usr/bin/env bash
#SBATCH --job-name=rules_h2h
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/users/kevlai/polyAI/.claude/worktrees/rules/runs/rules_h2h/out/%a.out
cd /home/users/kevlai/polyAI/.claude/worktrees/rules
export PATH=/home/users/kevlai/polyAI/.runtime/jdk-11.0.22+7/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FIX=0
# tasks 0-7: wrapper side rules OFF vs hook side rules ON; 8-15: the reverse (each side's path in both roles)
if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then export WRAP_RULES_OFF=all HOOK_RULES_OFF=; else export WRAP_RULES_OFF= HOOK_RULES_OFF=all; fi
export SHARD=$(( SLURM_ARRAY_TASK_ID % 8 ))
.venv/bin/python /home/users/kevlai/polyAI/.claude/worktrees/rules/runs/rules_h2h/rules_h2h.py /home/users/kevlai/polyAI/.claude/worktrees/league-us/runs/league_us_opponents/combat_pcritic_t1_league_swap_4m.cleanrl_model 50
