#!/bin/bash
# Dry run: validate scripts and zip logic without full pipeline
set -e
cd "$(dirname "$0")"

echo "=== 1. Shell syntax ==="
bash -n 02_run_track1.sh 02_run_track2.sh 02_run_track2_post.sh && echo "OK"

echo "=== 2. Source env + anon_suffix extraction ==="
[ -f env.sh ] && source env.sh 2>/dev/null || true
for cfg in configs/track1/anon_asrbn.yaml configs/track2/anon_ssl.yaml configs/track2/anon_post_ssl.yaml; do
  [ -f "$cfg" ] || continue
  suff=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f=open('$cfg'); print(load_hyperpyyaml(f,None).get('anon_suffix',''))")
  echo "  $cfg -> anon_suffix=$suff"
done

echo "=== 3. Zip exclusion test (Python) ==="
# Simulate zip -x "*libri*": paths to include vs exclude
python3 << 'EOF'
paths = [
    "exp/asv_ssl/results_ssl.csv",
    "exp/asv_ssl/ja_dev_enrolls_ssl-ja_dev_trials_f_ssl/scores.csv",
    "exp/asv_ssl/libri_dev_enrolls-libri_dev_trials_f/scores.csv",
]
excluded = [p for p in paths if "libri" in p]
included = [p for p in paths if "libri" not in p]
assert "libri" in excluded[0], "libri path should be excluded"
assert "ja_dev" in included[1], "track2 path should be included"
print("  libri excluded:", excluded)
print("  track2 included:", included)
print("  OK")
EOF

echo "=== 4. Config files exist ==="
for f in configs/track2/eval_post_{cn,ja,en,de,es,fr}.yaml configs/track2/eval_pre.yaml; do
  [ -f "$f" ] && echo "  $f" || echo "  MISSING: $f"
done

echo "=== 5. Zip command syntax (dry) ==="
anon_suffix=_ssl
results_exp=exp/results_summary/track2
mkdir -p "$results_exp" 2>/dev/null || true
touch "${results_exp}/result_for_rank${anon_suffix}" 2>/dev/null || true
# Run zip; if zip not installed, show the command we'd run
if command -v zip >/dev/null 2>&1; then
  zip ${results_exp}/result_for_submission${anon_suffix}.zip -r -x "*libri*" \
    "${results_exp}/result_for_rank${anon_suffix}" \
    exp/openai exp/ser_emotion2vec exp/asv_ssl 2>/dev/null || echo "  (zip produced empty/warning - paths may not exist yet)"
  [ -f "${results_exp}/result_for_submission${anon_suffix}.zip" ] && echo "  zip created" || echo "  zip skipped (paths missing)"
else
  echo "  zip not installed; command would be: zip -r -x '*libri*' ..."
fi

echo "=== Done ==="
