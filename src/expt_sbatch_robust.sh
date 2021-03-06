#!/bin/bash
#SBATCH --job-name=robust_test
#SBATCH --partition=defq
#SBATCH --time=03:00:00
#SBATCH --mem=8192
#SBATCH --output=../../slurm-job-outputs/rq2.2/res_%j.txt

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/myworkdir/disease_modeling/slurm-jobs.txt

module load python/3.7.3

cd
source ~/maxent-py37-pba/bin/activate
cd ~/myworkdir/disease_modeling/MaxEnt/src/
python -u rq2.2.py $1 $2 $3 $4 