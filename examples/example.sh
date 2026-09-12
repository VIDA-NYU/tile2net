#!/usr/bin/env bash

set -euo pipefail

# Prompt the user to enter the output directory
read -r -p "Please enter the output directory: " output_dir
output_dir="${output_dir/#\~/$HOME}"
mkdir -p "$output_dir"

echo "Tile generation will now begin."

# Run the published Boston example and retain viewable segmentation outputs.
location="42.35555189953313,-71.07168915322092,42.35364837213307,-71.06437423368418"

uv run python -m tile2net generate \
  --location "$location" \
  --output "$output_dir" \
  --name example \
  --dump_percent 100 \
  | uv run python -m tile2net inference \
      --local \
      --eval test \
      --dump_percent 100
