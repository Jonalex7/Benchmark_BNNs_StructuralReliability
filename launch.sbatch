#!/bin/bash
# Submission script for Hercules
#
#SBATCH --job-name=surrogate_test
#SBATCH --time=24:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem=5G               # Memory to allocate per allocated CPU core
#SBATCH --partition=batch
#
#SBATCH --mail-user=jmoran@uliege.be
#SBATCH --mail-type=ALL
source env/bin/activate
python main_train.py --method=dropout --lstate=four_branch