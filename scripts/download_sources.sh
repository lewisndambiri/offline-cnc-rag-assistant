#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/manuals

curl -L \
  -o data/raw/manuals/haas_mill_ngc_operator_2025.pdf \
  "https://www.haascnc.com/content/dam/haascnc/en/service/manual/operator/english---mill-ngc---operator%27s-manual---2025.pdf"

