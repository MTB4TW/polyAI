# Sourced by the training sbatch scripts: run ppo.py so it can stop cleanly and be continued (py_rl/cleanrl/cleanrl/resume.py).
#
# - The scripts ask SLURM for SIGUSR1 ten minutes before the time limit (#SBATCH --signal=B:USR1@600). The batch
#   shell forwards it to the trainer, which saves a checkpoint with its resume bundle at the end of the iteration.
# - RESUME_FROM=<run dir or checkpoint> continues a run exactly (ppo.py --resume).
# - AUTO_RESUME=1: after a time-limit stop (SIGUSR1), the job resubmits itself (TRAIN_SCRIPT) with RESUME_FROM set,
#   so a long run chains across the 48 h limit. A STOP file (runs/<run>/STOP) or a plain scancel stops for good.
# - An early stop exits with status 3, so jobs chained with --dependency=afterok (segments) never start from an
#   unfinished run. Use AUTO_RESUME for one long run; use afterok chains only for runs that finish normally.

run_trainer() {
    local args=("$@")
    if [ -n "${RESUME_FROM:-}" ]; then
        args+=(--resume "$RESUME_FROM")
        echo "Resuming from $RESUME_FROM"
    fi
    python "${args[@]}" &
    local pid=$!
    # shellcheck disable=SC2064  # expand $pid now
    trap "echo '[SIGNAL] SIGUSR1: asking the trainer to save and stop at the end of this iteration'; kill -USR1 $pid 2>/dev/null || true" USR1
    # shellcheck disable=SC2064
    trap "kill -TERM $pid 2>/dev/null || true" TERM
    local status=0
    while true; do
        set +e
        wait "$pid"
        status=$?
        set -e
        kill -0 "$pid" 2>/dev/null || break  # still running: wait was interrupted by a forwarded signal
    done
    trap - USR1 TERM
    local marker="logs/resume_${SLURM_JOB_ID:-none}.txt"
    # 3 = the trainer stopped before its last iteration and saved (py_rl/cleanrl/cleanrl/resume.py); the job keeps
    # that nonzero status so `afterok` dependents (e.g. chained segments) do not start from an unfinished run.
    if [ "$status" -eq 3 ] && [ -f "$marker" ]; then
        local run_dir reason
        run_dir=$(sed -n 1p "$marker")
        reason=$(sed -n 2p "$marker")
        echo "Trainer stopped early ($reason); continue with: RESUME_FROM=$run_dir sbatch ${TRAIN_SCRIPT:-<script>}"
        if [ "${AUTO_RESUME:-0}" = "1" ] && [ "$reason" = "SIGUSR1" ] && [ -n "${TRAIN_SCRIPT:-}" ]; then
            echo "AUTO_RESUME=1: resubmitting $TRAIN_SCRIPT"
            RESUME_FROM="$run_dir" sbatch "$TRAIN_SCRIPT"
        fi
    fi
    return "$status"
}
