#!/bin/bash

################################################################################
#
# Submit file for the self-driving batch job on Rosie.
#
# To submit your job, run 'sbatch <jobfile>'
# To view your jobs in the Slurm queue, run 'squeue -l -u <your_username>'
# To view details of a running job, run 'scontrol show jobid -d <jobid>'
# To cancel a job, run 'scancel <jobid>'
#
# See the manpages for salloc, srun, sbatch, squeue, scontrol, and scancel
# for more information or read the Slurm docs online: https://slurm.schedmd.com
#
################################################################################


# You _must_ specify the partition. Rosie's default is the 'teaching'
# partition for interactive nodes.  Another option is the 'batch' partition.
#SBATCH --partition=teaching

# The number of nodes to request
#SBATCH --nodes=1

# The number of GPUs to request
#SBATCH --gpus=1

# The number of CPUs to request per GPU
#SBATCH --cpus-per-gpu=1

# Kill the job if it takes longer than the specified time
# format: <days>-<hours>:<minutes>
#SBATCH --time=0-1:0


####
#
# Here's the actual job code.
# Note: You need to make sure that you execute this from the directory that
# model.py is located in OR provide an absolute path.
#
####

# Path to container
container="/data/containers/msoe-tensorflow.sif"

# Command to run inside container
command="python /home/harleys/CS-2300/lab8/model.py --data /data/cs2300/L8/ --h5modeloutput mAIprime_bs1024_e3.h5 --model commaAiModelPrime --batch_size 1024 --epochs 10"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}
