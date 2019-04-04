#!/bin/bash
#
#SBATCH --job-name=ninad-python
#SBATCH --partition=longq
#SBATCH --mem=4096
#SBATCH --time=02-02:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2
#SBATCH --output=test.out
#SBATCH --error=test.err

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

#export PATH=~/anaconda/bin:$PATH
#module purge
#module load python/2.7.12
#module load tensorflow/1.0.1
#module load cuda80/blas/8.0.44
#module load cuda80/fft/8.0.44
#module load cuda80/nsight/8.0.44
#module load cuda80/profiler/8.0.44
#module load cuda80/toolkit/8.0.44
#module load cudnn/7.0-cuda_8.0

#cd /home/ruppaal/Work/CV-Lab/Intelligent-Search

python topic_model.py
