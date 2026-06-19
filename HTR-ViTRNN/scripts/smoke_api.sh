#!/usr/bin/env bash
# Backend API Smoke Tests

set -euo pipefail

API_URL=${1:-"http://localhost:8000"}

echo "Running API smoke tests on ${API_URL}..."

# 1. Health check
echo -n "Checking GET /health: "
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/health")
if [[ "$HEALTH" == "200" ]]; then
  echo "OK (200)"
else
  echo "FAILED ($HEALTH)"
  exit 1
fi

# 2. Config check
echo -n "Checking GET /config: "
CONFIG=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/config")
if [[ "$CONFIG" == "200" ]]; then
  echo "OK (200)"
else
  echo "FAILED ($CONFIG)"
  exit 1
fi

# 3. Create dummy file and test upload job
echo "Creating dummy file to test job submission..."
DUMMY_IMG="/tmp/ocr_test_dummy.png"
# Create a 1x1 black pixel PNG using base64 representation of a PNG
echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" | base64 -d > "$DUMMY_IMG"

echo -n "Checking POST /api/jobs (Submit job): "
SUBMIT_RES=$(curl -s -F "file=@${DUMMY_IMG}" -F "mode=FAST" "${API_URL}/api/jobs")
JOB_ID=$(echo "$SUBMIT_RES" | grep -o '"job_id":"[^"]*' | grep -o '[^"]*$')

if [[ -n "$JOB_ID" ]]; then
  echo "OK (job_id: $JOB_ID)"
else
  echo "FAILED (Response: $SUBMIT_RES)"
  rm -f "$DUMMY_IMG"
  exit 1
fi

rm -f "$DUMMY_IMG"

# 4. Check job status polling
echo "Checking GET /api/jobs/${JOB_ID} (Poll status)..."
for i in {1..10}; do
  STATUS_RES=$(curl -s "${API_URL}/api/jobs/${JOB_ID}")
  STATUS=$(echo "$STATUS_RES" | grep -o '"status":"[^"]*' | grep -o '[^"]*$')
  PROGRESS=$(echo "$STATUS_RES" | grep -o '"progress":[0-9]*' | grep -o '[0-9]*$')
  
  echo "  Attempt $i: status = $STATUS, progress = $PROGRESS%"
  
  if [[ "$STATUS" == "done" ]]; then
    echo "Job finished processing successfully!"
    break
  elif [[ "$STATUS" == "failed" ]]; then
    echo "Job failed processing. Error details:"
    echo "$STATUS_RES"
    exit 1
  fi
  sleep 2
done

# 5. List jobs
echo -n "Checking GET /api/jobs (List jobs): "
LIST_JOBS=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/api/jobs")
if [[ "$LIST_JOBS" == "200" ]]; then
  echo "OK (200)"
else
  echo "FAILED ($LIST_JOBS)"
  exit 1
fi

echo "API SMOKE TESTS PASSED SUCCESSFULLY!"
