#!/bin/bash

HP_CONFIG="hp_config"
batch_size=(128)
for size in "${batch_size[@]}"; do
    cp "$HP_CONFIG.yml" "${HP_CONFIG}_batch_size_${size}.yml"
done
