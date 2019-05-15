#!/bin/bash
#
#SBATCH --job-name=nakred
#SBATCH --partition=longq
#SBATCH --mem=4096
#SBATCH --time=10-02:30:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --output=test-red.out
#SBATCH --error=test-red.err

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

export PATH=~/anaconda/bin:$PATH

#module purge
#module load python/2.7.12
#module load tensorflow/1.0.1
#module load cuda80/blas/8.0.44
#module load cuda80/fft/8.0.44
#module load cuda80/nsight/8.0.44
#module load cuda80/profiler/8.0.44
#module load cuda80/toolkit/8.0.44
#module load cudnn/7.0-cuda_8.0

source activate mxent
python main_mba_red.py
source deactivate
