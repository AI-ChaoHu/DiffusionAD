#!/bin/bash
##SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-4
#SBATCH --output=output/experiment-%A.%a.out
#SBATCH --error=output/error/experiment-%A.%a.out

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"

# make sure to change the environment
module load python/3.8
source ~/.virutalenvs/adbench/bin/activate

seed_arr=(0 1 2 3 4)
seed_len=${#seed_arr[@]}


for ((idx_1=0; idx_1<$seed_len; idx_1++))
do
    task_id=`expr $idx_1`

    #check for the correct task id
    if [ $task_id == $SLURM_ARRAY_TASK_ID ]
    then
        echo "Seed: ${seed_arr[$idx_1]}"
        srun python -u run_diffusion.py --seed ${seed_arr[$idx_1]}
    fi
done