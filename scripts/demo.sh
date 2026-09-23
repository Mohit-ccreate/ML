#!/usr/bin/env bash
# One-command OptionEdge demo bring-up (for the sponsor presentation).
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

if [[ ! -f artifacts/metrics/holdout_results.json || ! -f artifacts/metrics/walkforward_results.json ]]; then
  echo "==> Artifacts missing — running full pipeline first"
  python3 scripts/run_pipeline.py
else
  echo "==> Artifacts present — skipping training (delete artifacts/ to force retrain)"
fi

echo "==> Unit tests"
python3 tests/test_basic.py

echo "==> Starting dashboard on 0.0.0.0:8501"
exec streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false
