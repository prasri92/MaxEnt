#!/bin/bash
#SBATCH --job-name=maxent_true_dist
#SBATCH --partition=defq
#SBATCH --time=03:00:00
#SBATCH --mem=8192
#SBATCH --output=res_%j.txt

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module load python/3.7.3

cd
source ~/maxent-py37-pba/bin/activate
cd MaxEnt/src/codebase
python synthetic_data.py $1 