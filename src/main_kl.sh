#!/bin/bash
#SBATCH --job-name=mxent_synth_v4
#SBATCH --partition=defq
#SBATCH --time=05:00:00
#SBATCH --mem=8192
#SBATCH --output=res_%j.txt

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module load python/3.7.3

export PATH=~/maxent-py37-pba/bin:$PATH

cd 
source ~/maxent-py37-pba/bin/activate
cd MaxEnt/src
python -u main_kl_test.py $1
