#!/bin/bash

HP_CONFIG="hp_config.yml"
OUTPUT_PREFIX="hp_config"

batch_sizes=(2 64 128)

for bs in "${batch_sizes[@]}"; do
    new_file="${OUTPUT_PREFIX}_batch_size_${bs}.yml"

    # Copy and edit batch_size
    cp "$HP_CONFIG" "$new_file"

    # Replace the batch_size line
    sed -i "s/^batch_size:.*/batch_size: ${bs}/" "$new_file"

    echo "Created $new_file with batch_size=${bs}"
done
