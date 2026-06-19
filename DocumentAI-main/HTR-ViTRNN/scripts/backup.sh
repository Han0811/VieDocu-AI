#!/usr/bin/env bash
# Backup database and runs directory

set -euo pipefail

BACKUP_DIR="./backups"
RUNS_DIR="./backend_runs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PATH="${BACKUP_DIR}/backup_${TIMESTAMP}"

mkdir -p "${BACKUP_DIR}"

echo "Starting system backup..."

# 1. Backup SQLite Database if it exists
if [[ -f "${RUNS_DIR}/jobs.sqlite3" ]]; then
  echo "Backing up SQLite database..."
  cp "${RUNS_DIR}/jobs.sqlite3" "${BACKUP_PATH}.sqlite3"
fi

# 2. Archive runs outputs (excluding raw inputs or tmp page files if desired, or archive whole directory)
if [[ -d "${RUNS_DIR}" ]]; then
  echo "Archiving runs directory (excluding caches)..."
  tar --exclude='pages' --exclude='*.zip' -czf "${BACKUP_PATH}_runs.tar.gz" -C "${RUNS_DIR}" .
fi

echo "Backup completed successfully! Files saved:"
ls -lh "${BACKUP_PATH}"*

# Keep only the last 10 backups to prevent disk bloat
echo "Pruning old backups..."
find "${BACKUP_DIR}" -type f \( -name "*.sqlite3" -o -name "*.tar.gz" \) -mtime +10 -delete
