#!/bin/bash
#SBATCH --job-name=sup_gat
#SBATCH --partition=defq
#SBATCH --time=10:00:00
#SBATCH --mem=8192
#SBATCH --output=../../slurm-job-outputs/res_%j.txt

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/myworkdir/disease_modeling/slurm-jobs.txt

module load python/3.7.3
# module load gurobi/811

cd
source ~/maxent-py37-pba/bin/activate
cd ~/myworkdir/disease_modeling/MaxEnt/src/
python -u vis_rq5.1.py 