#!/bin/sh

# Name of project folder
PROJECT_NAME=fab-torch
# Original code folder is here
MAIN_DIR=/home/tbbakke/${PROJECT_NAME}
# Launch dir
LAUNCH_DIR=${MAIN_DIR}/launch
mkdir -p "${LAUNCH_DIR}"


JOB_NAME=0221_hyperparams

# Create dir for specific experiment run
dt=$(date '+%F_%T.%3N')
LOGS_DIR=${LAUNCH_DIR}/${dt}
mkdir -p "${LOGS_DIR}"
# Copy code to experiment folder
rsync -arm ${MAIN_DIR} --stats --exclude-from=${MAIN_DIR}/"SYNC_EXCLUDE" ${LOGS_DIR};
cd ${LOGS_DIR}/${PROJECT_NAME}
echo "Launching job from: ${PWD}"

# Make SLURM file
JOB_NAME=${JOB_NAME}
SLURM=${LOGS_DIR}/run.slrm
echo "${SLURM}"
echo "#!/bin/bash" > ${SLURM}
echo "#SBATCH --job-name=$JOB_NAME" >> ${SLURM}
echo "#SBATCH --output=${LOGS_DIR}/%j.out" >> ${SLURM}
echo "#SBATCH --error=${LOGS_DIR}/%j.err" >> ${SLURM}
echo "#SBATCH --gres=gpu:1" >> ${SLURM}
echo "#SBATCH --cpus-per-task=12" >> ${SLURM}
echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
echo "#SBATCH --mem=8G" >> ${SLURM}
echo "#SBATCH --time=0-2:00:00" >> ${SLURM}
echo "#SBATCH --nodes=1" >> ${SLURM}
echo "export PYTHONPATH=:\$PYTHONPATH:" >> ${SLURM}
{
    echo HYDRA_FULL_ERROR=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 /home/tbbakke/anaconda3/envs/fab/bin/python \
        PYTHONPATH=${LOGS_DIR}/${PROJECT_NAME} ${LOGS_DIR}/${PROJECT_NAME}/experiments/solvation/run.py \
        --flow.blocks 12 --flow.hidden_units 256 --flow.num_bins 9 \
        --training.n_iterations 500 --evaluation.n_eval 50 --evaluation.n_plots 10 --evaluation.n_checkpoints 1
} >> ${SLURM}

sbatch ${SLURM}
