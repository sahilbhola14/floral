#!/bin/bash

# -------------------------------
# Sweep space
# -------------------------------
# FLORAL_VALUES=(True False)
# N_TRAIN_SAMPLES_VALUES=(3500 1000 500 10)
# OBJECTIVES=("fno" "fm")

FLORAL_VALUES=(True False)
N_TRAIN_SAMPLES_VALUES=(3500)
OBJECTIVES=("fno" "fm")

CONFIG="config.yml"
HP_CONFIG="hp_config.yml"
LOG_FILE="ablation_submissions.log"

# -------------------------------
# Loop over all combinations
# -------------------------------
for floral in "${FLORAL_VALUES[@]}"; do
  for n_train in "${N_TRAIN_SAMPLES_VALUES[@]}"; do
    for objective in "${OBJECTIVES[@]}"; do

    EXP_NAME="floral_${floral}_ntrain_${n_train}_obj_${objective}"

    # One single override string
    OVERRIDES="config.floral=${floral} config.dataloader.n_train_samples=${n_train} config.train.objective=${objective}"

    # Timestamp for logging
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    echo "Submitting: ${EXP_NAME}"
    echo "Overrides: ${OVERRIDES}"

    # Log submission
    echo "[${TIMESTAMP}] job_name=${EXP_NAME} overrides=${OVERRIDES}" >> "${LOG_FILE}"

    # Submit job
    sbatch \
    --job-name="${EXP_NAME}" \
    submit.sh \
    --config "${CONFIG}" \
    --hp_config "${HP_CONFIG}" \
    ${OVERRIDES}

    done
  done
done
