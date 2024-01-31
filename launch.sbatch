#!/bin/bash
# Submission script for nic5
#
#SBATCH --job-name=bnn_surrogate_test
#SBATCH --time=36:00:00 # hh:mm:ss
#SBATCH --array=0-5      #for the number of arguments to be tested
#SBATCH --ntasks=1
#SBATCH --mem=10G        # Memory to allocate per allocated CPU core
#SBATCH --partition=batch
#Define an array of arguments
ARGS=("--method=bnnbpp --lstate=parabolic" "--method=dropout --lstate=parabolic" "--method=sghmc --lstate=himmelblau" "--method=sghmc --lstate=parabolic" "--method=sghmc --lstate=electric" "--method=sghmc --lstate=high_dim")
# Use SLURM_ARRAY_TASK_ID to get the correct argument
ARG=${ARGS[$SLURM_ARRAY_TASK_ID]}
#
#SBATCH --mail-user=jmoran@uliege.be
#SBATCH --mail-type=ALL
source env/bin/activate
python main_train.py $ARG
