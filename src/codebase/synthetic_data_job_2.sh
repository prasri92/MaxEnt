#!/bin/bash
#SBATCH --job-name=maxent_true_dist
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --mem=8192
#SBATCH --output=../../../slurm-job-outputs/syn_large_data/res_%j.txt

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/myworkdir/disease_modeling/slurm-jobs.txt

module load python/3.7.3

cd
source ~/maxent-py37-pba/bin/activate
cd ~/myworkdir/disease_modeling/MaxEnt/src/codebase
python -u synthetic_data_3.py $1 $2  