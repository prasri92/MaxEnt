#!/bin/bash
#SBATCH --job-name=mxent_synth_v4
#SBATCH --partition=defq
#SBATCH --time=10:00:00
#SBATCH --mem=9216
#SBATCH --output=syn_maxent_%j.txt

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module load python/3.7.3

export PATH=~/maxent-py37-pba/bin:$PATH

cd 
source ~/maxent-py37-pba/bin/activate
cd MaxEnt/src
python -u main_maxent_perturb.py $1 $2
