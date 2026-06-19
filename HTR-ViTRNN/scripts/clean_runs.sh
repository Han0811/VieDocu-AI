#!/usr/bin/env bash
# Prune old job runs (default: older than 7 days) to free up disk space.

set -euo pipefail

RUNS_DIR="./backend_runs"
DAYS_TO_KEEP=${1:-7}

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "Runs directory '${RUNS_DIR}' does not exist. Nothing to clean."
  exit 0
fi

echo "Cleaning up runs older than ${DAYS_TO_KEEP} days in ${RUNS_DIR}..."

# Find folders inside backend_runs (excluding metadata files) and delete them
# We check modification time of folders or files inside.
# To be safe and clean, we iterate over job directories matching pattern YYYYMMDD_*
find "${RUNS_DIR}" -maxdepth 1 -mindepth 1 -type d -name "20[0-9][0-9][0-9][0-9][0-9][0-9]_*" -mtime +"${DAYS_TO_KEEP}" | while read -r job_dir; do
  job_id=$(basename "${job_dir}")
  echo "Deleting expired job runs directory: ${job_id}"
  rm -rf "${job_dir}"
done

# We also clean any stray ZIP files in backend_runs/zips
if [[ -d "${RUNS_DIR}/zips" ]]; then
  find "${RUNS_DIR}/zips" -type f -name "*.zip" -mtime +"${DAYS_TO_KEEP}" -delete
fi

echo "Pruning database records for deleted runs..."
# We can run a quick SQLite query if sqlite3 CLI is installed
if command -v sqlite3 &>/dev/null && [[ -f "${RUNS_DIR}/jobs.sqlite3" ]]; then
  sqlite3 "${RUNS_DIR}/jobs.sqlite3" "DELETE FROM jobs WHERE status != 'processing' AND updated_at < datetime('now', '-${DAYS_TO_KEEP} days');"
  echo "Database records pruned."
fi

echo "Runs clean-up complete."
