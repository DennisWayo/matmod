#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-macstudio}"
REMOTE_DIR="${REMOTE_DIR:-~/matmod}"
REMOTE_PYTHON="${REMOTE_PYTHON:-~/miniforge3/envs/gpaw-tddft-legacy/bin/python}"
REMOTE_RESULTS_REL="undergrads/lucas/results"
REMOTE_JOBS_REL="${REMOTE_RESULTS_REL}/jobs"
REMOTE_ANALYSIS_CMD="${REMOTE_PYTHON} undergrads/lucas/analyze_results.py && ${REMOTE_PYTHON} undergrads/lucas/export_qml_dataset.py && ${REMOTE_PYTHON} undergrads/lucas/analysis_co2_reduction.py && ${REMOTE_PYTHON} undergrads/lucas/render_publication_figures.py"
JOB_NAMES=(pipeline dft tddft pathways analysis)

usage() {
  cat <<'EOF'
Usage:
  undergrads/lucas/remote_macstudio.sh push
  undergrads/lucas/remote_macstudio.sh check-env
  undergrads/lucas/remote_macstudio.sh run [pipeline flags]
  undergrads/lucas/remote_macstudio.sh run-bg [pipeline flags]
  undergrads/lucas/remote_macstudio.sh run-dft [dft flags]
  undergrads/lucas/remote_macstudio.sh run-dft-bg [dft flags]
  undergrads/lucas/remote_macstudio.sh run-tddft [tddft flags]
  undergrads/lucas/remote_macstudio.sh run-tddft-bg [tddft flags]
  undergrads/lucas/remote_macstudio.sh run-pathways [pathway flags]
  undergrads/lucas/remote_macstudio.sh run-pathways-bg [pathway flags]
  undergrads/lucas/remote_macstudio.sh run-analysis
  undergrads/lucas/remote_macstudio.sh run-analysis-bg
  undergrads/lucas/remote_macstudio.sh status [pipeline|dft|tddft|pathways|analysis|all]
  undergrads/lucas/remote_macstudio.sh tail-log [pipeline|dft|tddft|pathways|analysis]
  undergrads/lucas/remote_macstudio.sh stop [pipeline|dft|tddft|pathways|analysis|all]
  undergrads/lucas/remote_macstudio.sh cleanup-local
  undergrads/lucas/remote_macstudio.sh cleanup-remote
  undergrads/lucas/remote_macstudio.sh pull-light

Environment overrides:
  REMOTE_HOST   (default: macstudio)
  REMOTE_DIR    (default: ~/matmod)
  REMOTE_PYTHON (default: ~/miniforge3/envs/gpaw-tddft-legacy/bin/python)
EOF
}

push_repo() {
  rsync -az \
    --exclude '.git/' \
    --exclude '.idea/' \
    --exclude 'undergrads/lucas/results/' \
    --exclude 'undergrads/lucas/results/dft/' \
    --exclude 'undergrads/lucas/results/tddft/' \
    "${LOCAL_REPO_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
  echo "[remote-sync] pushed repository to ${REMOTE_HOST}:${REMOTE_DIR}"
}

check_env() {
  ssh "${REMOTE_HOST}" "${REMOTE_PYTHON} -c 'import numpy,ase,gpaw; print(\"env-ok\")'"
}

run_remote_command() {
  local command="$1"
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && ${command}"
}

run_remote_python() {
  local script_name="$1"
  shift || true
  local flags="$*"
  run_remote_command "${REMOTE_PYTHON} undergrads/lucas/${script_name} ${flags}"
}

run_remote_python_bg() {
  local job_name="$1"
  local script_name="$2"
  shift 2 || true
  local flags="$*"
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && mkdir -p ${REMOTE_JOBS_REL} && nohup ${REMOTE_PYTHON} -u undergrads/lucas/${script_name} ${flags} > ${REMOTE_JOBS_REL}/${job_name}.log 2>&1 < /dev/null & echo \$! > ${REMOTE_JOBS_REL}/${job_name}.pid && cat ${REMOTE_JOBS_REL}/${job_name}.pid"
}

run_remote_analysis() {
  run_remote_command "${REMOTE_ANALYSIS_CMD}"
}

run_remote_analysis_bg() {
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && mkdir -p ${REMOTE_JOBS_REL} && nohup bash -lc '${REMOTE_ANALYSIS_CMD}' > ${REMOTE_JOBS_REL}/analysis.log 2>&1 < /dev/null & echo \$! > ${REMOTE_JOBS_REL}/analysis.pid && cat ${REMOTE_JOBS_REL}/analysis.pid"
}

status_remote_job() {
  local job_name="$1"
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && pid_file=${REMOTE_JOBS_REL}/${job_name}.pid; if [[ -f \$pid_file ]]; then pid=\$(cat \$pid_file 2>/dev/null || true); if [[ -n \$pid ]] && ps -p \$pid >/dev/null 2>&1; then ps -p \$pid -o pid,pcpu,pmem,etime,command; else echo '${job_name}: stale or exited pid'; fi; else echo '${job_name}: no pid file'; fi"
}

status_remote() {
  local target="${1:-all}"
  if [[ "${target}" == "all" ]]; then
    local job
    for job in "${JOB_NAMES[@]}"; do
      status_remote_job "${job}"
    done
    return
  fi
  status_remote_job "${target}"
}

tail_remote_log() {
  local job_name="${1:-pipeline}"
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && log_file=${REMOTE_JOBS_REL}/${job_name}.log; if [[ -f \$log_file ]]; then tail -n 80 \$log_file; else echo '${job_name}: no log file'; fi"
}

stop_remote_job() {
  local job_name="$1"
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && pid_file=${REMOTE_JOBS_REL}/${job_name}.pid; if [[ -f \$pid_file ]]; then pid=\$(cat \$pid_file 2>/dev/null || true); if [[ -n \$pid ]] && ps -p \$pid >/dev/null 2>&1; then kill \$pid; fi; rm -f \$pid_file; echo '${job_name}: stopped'; else echo '${job_name}: no pid file'; fi"
}

stop_remote() {
  local target="${1:-all}"
  if [[ "${target}" == "all" ]]; then
    local job
    for job in "${JOB_NAMES[@]}"; do
      stop_remote_job "${job}"
    done
    return
  fi
  stop_remote_job "${target}"
}

cleanup_local_results() {
  local results_dir="${LOCAL_REPO_DIR}/undergrads/lucas/results"
  mkdir -p "${results_dir}"

  while IFS= read -r pid_file; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      continue
    fi
    rm -f "${pid_file}"
  done < <(find "${results_dir}" -maxdepth 2 -type f -name '*.pid' | sort)

  rm -f \
    "${results_dir}/dft_summary_before_ceo2_fix.csv" \
    "${results_dir}/full_dft.log" \
    "${results_dir}/remote_dft_smoke.log"

  find "${results_dir}" -type f -name '*.error.txt' -delete
  echo "[cleanup-local] removed stale pid files and obsolete failed logs"
}

cleanup_remote_results() {
  ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && mkdir -p ${REMOTE_JOBS_REL} && \
    find ${REMOTE_RESULTS_REL} -maxdepth 2 -type f -name '*.pid' | while read -r pid_file; do \
      pid=\$(cat \"\$pid_file\" 2>/dev/null || true); \
      if [[ -n \"\$pid\" ]] && ps -p \"\$pid\" >/dev/null 2>&1; then \
        continue; \
      fi; \
      rm -f \"\$pid_file\"; \
    done && \
    rm -f ${REMOTE_RESULTS_REL}/full_dft.log ${REMOTE_RESULTS_REL}/remote_dft_smoke.log ${REMOTE_RESULTS_REL}/dft_summary_before_ceo2_fix.csv && \
    find ${REMOTE_RESULTS_REL} -type f -name '*.error.txt' -delete && \
    rm -f ${REMOTE_RESULTS_REL}/dft/_reference/co2_ref.gpw ${REMOTE_RESULTS_REL}/dft/_reference/co2_ref.opt.log ${REMOTE_RESULTS_REL}/dft/_reference/co2_ref.traj ${REMOTE_RESULTS_REL}/dft/_reference/co2_ref.txt ${REMOTE_RESULTS_REL}/dft/_reference/co2_reference.json && \
    echo '[cleanup-remote] removed stale pid files and obsolete failed logs'"
}

pull_light_outputs() {
  rsync -az \
    --include='undergrads/' \
    --include='undergrads/lucas/' \
    --include='undergrads/lucas/results/' \
    --include='undergrads/lucas/results/analysis/' \
    --include='undergrads/lucas/results/analysis/***' \
    --include='undergrads/lucas/results/tddft/' \
    --include='undergrads/lucas/results/tddft/*/' \
    --include='undergrads/lucas/results/tddft/*/*/' \
    --include='undergrads/lucas/results/tddft/*/*/transitions.csv' \
    --include='undergrads/lucas/results/jobs/' \
    --include='undergrads/lucas/results/*.csv' \
    --include='undergrads/lucas/results/*.md' \
    --include='undergrads/lucas/results/*.json' \
    --include='undergrads/lucas/results/*.log' \
    --include='undergrads/lucas/results/*.pid' \
    --include='undergrads/lucas/results/jobs/*.log' \
    --include='undergrads/lucas/results/jobs/*.pid' \
    --exclude='*' \
    "${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_REPO_DIR}/"
  echo "[remote-sync] pulled lightweight results into local undergrads/lucas/results"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="$1"
  shift || true

  case "${cmd}" in
    push)
      push_repo
      ;;
    check-env)
      check_env
      ;;
    run)
      run_remote_python "run_pipeline.py" "$@"
      ;;
    run-bg)
      run_remote_python_bg "pipeline" "run_pipeline.py" "$@"
      ;;
    run-dft)
      run_remote_python "run_dft.py" "$@"
      ;;
    run-dft-bg)
      run_remote_python_bg "dft" "run_dft.py" "$@"
      ;;
    run-tddft)
      run_remote_python "run_tddft.py" "$@"
      ;;
    run-tddft-bg)
      run_remote_python_bg "tddft" "run_tddft.py" "$@"
      ;;
    run-pathways)
      run_remote_python "run_co2rr_pathways.py" "$@"
      ;;
    run-pathways-bg)
      run_remote_python_bg "pathways" "run_co2rr_pathways.py" "$@"
      ;;
    run-analysis)
      run_remote_analysis
      ;;
    run-analysis-bg)
      run_remote_analysis_bg
      ;;
    status)
      status_remote "${1:-all}"
      ;;
    tail-log)
      tail_remote_log "${1:-pipeline}"
      ;;
    stop)
      stop_remote "${1:-all}"
      ;;
    cleanup-local)
      cleanup_local_results
      ;;
    cleanup-remote)
      cleanup_remote_results
      ;;
    pull-light)
      pull_light_outputs
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
