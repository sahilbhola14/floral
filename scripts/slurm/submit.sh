#!/bin/bash
#SBATCH --job-name=onedcorr          # job name
#SBATCH --mail-user=sbhola@umich.edu    # email
#SBATCH --mail-type=BEGIN,FAIL,END      # mail type
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # tasks per node
#SBATCH --cpus-per-task=10               # cpu per task
#SBATCH --gpus-per-node=h100:1          # gpus per node
#SBATCH --mem=10g                       # total memory
#SBATCH --time=10:00:00                    # time limite: hrs:min:sec
#SBATCH --account=kdur                  # account to charge
#SBATCH --partition=project_l           # Partition
#SBATCH --nice=0 # Optional Priority

export LOGLEVEL=INFO                    # Log information
source ~/.zshrc                         # Source the shell
conda init                              # Conda initialization
conda activate floral                   # Load Conda environment

# -------------------------------
# Metadata
# -------------------------------
JOB_ID=${SLURM_JOB_ID}
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Create log directory
LOG_DIR=logs
mkdir -p ${LOG_DIR}

# Log files
RUN_LOG=${LOG_DIR}/run_${JOB_ID}.log
META_LOG=${LOG_DIR}/meta_${JOB_ID}.log

# -------------------------------
# Log submission details
# -------------------------------
echo "==============================" >> ${META_LOG}
echo "Job ID       : ${JOB_ID}" >> ${META_LOG}
echo "Submit Time  : ${TIMESTAMP}" >> ${META_LOG}
echo "Node List    : ${SLURM_NODELIST}" >> ${META_LOG}
echo "Num Tasks    : ${SLURM_NTASKS}" >> ${META_LOG}
echo "Working Dir  : $(pwd)" >> ${META_LOG}
echo "------------------------------" >> ${META_LOG}
echo "Command      : python run.py $@" >> ${META_LOG}
echo "Overrides    : $@" >> ${META_LOG}
echo "==============================" >> ${META_LOG}

# Also store job id separately (optional)
echo ${JOB_ID} > job_id

# -------------------------------
# Run job
# -------------------------------
START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Run started at ${START_TIME}" >> ${META_LOG}

srun python run.py "$@" > ${RUN_LOG} 2>&1

END_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Run finished at ${END_TIME}" >> ${META_LOG}
echo "==============================" >> ${META_LOG}
