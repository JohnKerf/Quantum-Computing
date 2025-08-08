#!/bin/bash -l

#SBATCH --job-name=QM_VQE          # Job name
#SBATCH --output=VQE_DE_%j.out    # Standard output file (%j is replaced by job ID)
#SBATCH --error=VQE_DE_%j.err     # Standard error file (%j is replaced by job ID)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=100       # Number of CPU cores per task
#SBATCH --time=24:00:00           # Time limit (HH:MM:SS)
#SBATCH --partition=nodes         # Specify the partition/queue
#SBATCH --export=ALL              # Export environment variables

# Activate your Conda environment
source /opt/apps/pkg/tools/miniforge3/25.3.0_python3.12.10/etc/profile.d/conda.sh
conda activate /users/johnkerf/.conda/envs/QC

export OMP_NUM_THREADS=1           # OpenMP-based threading
export MKL_NUM_THREADS=1           # Intel MKL threading
export NUMEXPR_NUM_THREADS=1       # NumExpr threading
export OPENBLAS_NUM_THREADS=1      # OpenBLAS threading

# Run your Python script
python /users/johnkerf/Quantum\ Computing/SUSY-QM/EnergySampling/Sample.py

