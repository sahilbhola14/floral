#!/bin/bash

# -------------------------------
# Sweep space
# -------------------------------
# FLORAL_VALUES=(True False)
# N_TRAIN_SAMPLES_VALUES=(3500 1000 500 10)
# OBJECTIVES=("fno" "fm")

FLORAL_VALUES=(True False)
N_UNIQUE_TRAIN_CONDITIONS=(2 4 10 20 40) # same size as BATCH_SIZE
BATCH_SIZE=(32 32 32 32 32) # same size as N_TRAIN_SAMPLES_VALUES
OBJECTIVES=("fno" "fm")
WANDB_GROUP="vary_training_size"

BASE_CONFIG="config"
BASE_HP_CONFIG="hp_config"
LOG_FILE="ablation_submissions.log"

# -------------------------------
# Loop over all combinations
# -------------------------------
for i in "${!N_UNIQUE_TRAIN_CONDITIONS[@]}"; do
    n_unique_train=${N_UNIQUE_TRAIN_CONDITIONS[$i]} # extract the train samples
    batch_size=${BATCH_SIZE[$i]} # extact hte batch size
    for floral in "${FLORAL_VALUES[@]}"; do
    for objective in "${OBJECTIVES[@]}"; do

    EXP_NAME="floral_${floral}_n_unique_train_${n_unique_train}_obj_${objective}"

    # One single override string
    OVERRIDES="wandb_group=${WANDB_GROUP} floral=${floral} dataloader.n_unique_train_conditions=${n_unique_train} train.objective=${objective}"

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
    --config "${BASE_CONFIG}.yml" \
    --hp_config "${BASE_HP_CONFIG}_batch_size_${batch_size}.yml" \
    ${OVERRIDES}

    done
  done
done
