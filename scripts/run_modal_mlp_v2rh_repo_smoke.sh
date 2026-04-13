#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_dir="$repo_root/artifacts/modal_repo_smoke"
mkdir -p "$artifact_dir"

ts="$(date +%Y%m%d_%H%M%S)"
log_file="$artifact_dir/run_${ts}.log"
json_file="$artifact_dir/run_${ts}.json"
latest_log="$artifact_dir/latest.log"
latest_json="$artifact_dir/latest.json"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -f "$repo_root/smoke_data/v4/shard_000/train_input.h5" ]]; then
  "$PYTHON_BIN" "$repo_root/scripts/create_synthetic_climsim_smoke_data.py" \
    --output-dir "$repo_root/smoke_data/v4"
fi

(
  cd "$repo_root"
  modal run scripts/modal_train_climsim_repo.py --expname mlp_v2rh_modal_repo_smoke
) | tee "$log_file"

python3 - <<'PY' "$log_file" "$json_file" "$latest_json"
from pathlib import Path
import json, sys
log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
latest_json = Path(sys.argv[3])
text = log_path.read_text()
start = text.find('{')
end = text.rfind('}')
if start == -1 or end == -1 or end < start:
    raise SystemExit('Could not find JSON payload in modal output log')
payload = json.loads(text[start:end+1])
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY

cp "$log_file" "$latest_log"

python3 - <<'PY' "$latest_json"
from pathlib import Path
import json, sys
payload = json.loads(Path(sys.argv[1]).read_text())
assert payload['train_exit_code'] == 0, payload
assert payload['eval_exit_code'] == 0, payload
for key in ['rmse', 'mae', 'mse', 'r2']:
    assert key in payload['eval'], payload
print(json.dumps(payload['eval'], indent=2, sort_keys=True))
PY
