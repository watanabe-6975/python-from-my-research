#!/usr/bin/env bash
# 全章を上から順に実行して、実行済みのノートブックを notebooks/executed/ に保存する。
# 使い方：bash tools/run_all.sh          （data/ の計算結果があれば使う）
#         bash tools/run_all.sh --fresh  （data/*.npz を消して、計算からやり直す）
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/build_notebooks.py
if [[ "${1:-}" == "--fresh" ]]; then rm -f data/*.npz; fi
mkdir -p notebooks/executed
fail=0
for nb in notebooks/*.ipynb; do
  start=$(date +%s)
  if jupyter nbconvert --to notebook --execute "$nb" --output-dir notebooks/executed \
       --ExecutePreprocessor.timeout=600 >/dev/null 2>/tmp/nbconvert_err.log; then
    echo "OK   $(( $(date +%s) - start ))s  $nb"
  else
    echo "FAIL $nb"; tail -20 /tmp/nbconvert_err.log; fail=1
  fi
done
exit $fail
