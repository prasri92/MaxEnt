#!/bin/bash
#SBATCH --job-name=maxent_18_20
#SBATCH --partition=defq
#SBATCH --time=10:00:00
#SBATCH --mem=8192
#SBATCH --output=../../../synthetic_data/gen_large/res_%j.txt

# Log what we're running and where.
# echo $SLURM_JOBID - `hostname` >> ~/myworkdir/disease_modeling/slurm-jobs.txt

module load python/3.7.3

cd
source ~/maxent-py37-pba/bin/activate
cd ~/myworkdir/disease_modeling/MaxEnt/src/codebase
python -u synthetic_data_5.py $1 $2 $3