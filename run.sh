#!/bin/bash
## Set a job name for this job. This name will show in the queue
#SBATCH --job-name=DDM_onestage

## Set parameters for this job, write each parameter starting with "#SBATCH"
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task = 2

## Set output files.
## ".out" file has everything you printed with the standard output
## ".err" file has the error messages
## "%A" means the job id, and "%a" means the job array index
#SBATCH -e ./stderr/slurm-%A_%a.err
#SBATCH -o ./stdout/slurm-%A_%a.out

## Add this line to submit a job array, here it's 20 jobs indexed from 0 to 19
#SBATCH --array=0-2


case $SLURM_ARRAY_TASK_ID in
    0)
        LOWER_BOUNDS='[0.0005, 0.0001, 0.2, 0.8, 8, 4]'
        UPPER_BOUNDS='[0.005, 0.005, 0.6, 1.0, 15, 10]'
        ;;
    1)
        LOWER_BOUNDS='[0.00001, 0.00002, 0.2, 0.6, 9, 3]'
        UPPER_BOUNDS='[0.0004, 0.0005, 0.7, 1.1, 17, 11]'
        ;;
    2)
        LOWER_BOUNDS='[0.0004, 0.0002, 0.3, 0.8, 9, 5]'
        UPPER_BOUNDS='[0.001, 0.001, 0.8, 1.2, 17, 11]'
        ;;
esac


export LOWER_BOUNDS=$LOWER_BOUNDS
export UPPER_BOUNDS=$UPPER_BOUNDS


julia -p 2 two_stage/main.jl --run_num $SLURM_ARRAY_TASK_ID
