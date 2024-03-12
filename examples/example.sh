#!/bin/bash

# Prompt the user to enter the output directory
echo "Please enter the output directory:"
read output_dir
echo "Tile generation will now begin."

# Run the generate and inference commands
#location="42.35555189953313, -71.07168915322092, 42.35364837213307, -71.06437423368418"
location='Washington Square Park, New York, NY'
python -m tile2net generate -l "$location" -o "$output_dir" -n example | python -m tile2net inference
