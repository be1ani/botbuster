#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG="$ROOT/docs/images"
mkdir -p "$IMG"
DIAG="$ROOT/docs/diagrams"

run_mmdc() {
  local in="$1"
  local out="$2"
  local pconf="$ROOT/docs/puppeteer.json"
  if command -v mmdc >/dev/null 2>&1; then
    mmdc -i "$in" -o "$out" -b transparent -p "$pconf"
  else
    npx --yes @mermaid-js/mermaid-cli -i "$in" -o "$out" -b transparent -p "$pconf"
  fi
}

for f in "$DIAG"/*.mmd; do
  [[ -e "$f" ]] || continue
  base=$(basename "$f" .mmd)
  echo "Rendering $f -> $IMG/${base}.png"
  run_mmdc "$f" "$IMG/${base}.png"
done
echo "Done. PNG files are in $IMG"
