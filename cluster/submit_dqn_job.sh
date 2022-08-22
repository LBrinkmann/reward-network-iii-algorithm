#!/bin/bash

# This script parallelizes the jobs related to the Reward Networks project on the MPIB Cluster (tardis)
# The process is executed on tardis using a job wrapper for SLURM resource management system.
# 
# Overview of methods:
# - dqn: calls DQN agent to be trained on solving reward networks 
#
# PROJECT: Reward Networks III
# author @ Sara Bonati
# Center for Humans and Machines - Max Planck Institute for Human Development - Berlin
#-----------------------------------------------------------------------------

# ---------------------------Worfklow settings--------------------------------
method="dqn" 

# -------Directory and filename (fn) management-------------------------------
# roots
home_dir="/mnt/beegfs/home/bonati"
project_dir="${home_dir}/CHM/reward_network"
code_dir="${project_dir}/reward-network-iii-algorithm"
data_dir="${project_dir}/data"

# script path
main_script_fn=${code_dir}/notebooks/dqn_agent.py

# output parent directories
if [[ "${method}" == "dqn" ]]; then
    output_dir="${data_dir}/solutions"
    log_dir="${project_dir}/logs"
    main_script_fn="${code_dir}/kamel.py"

# Create directories if non-existent
if [[ ! -d "${output_dir}" ]]; then
  mkdir -p "${output_dir}"
fi
if [ ! -d "${log_dir}" ]; then
  mkdir -p "${log_dir}"
fi
if [ ! -d "${log_dir}/${method}" ]; then
  mkdir -p "${log_dir}/${method}"
fi

# -------Define computation parameters------------------------
n_cpus=1 # maximum number of cpus per process
mem=7GB # memory demand
device="gpu"
cuda_specs="gpu:pascal:1"

if [[ "${method}" == "dqn" ]]; then

  # Create slurm job file
  echo "#!/bin/bash" >job.slurm
  # Name job file
  echo "#SBATCH --job-name reward_networks-${method}" >>job.slurm
  # Specify maximum run time
  echo "#SBATCH --time 24:00:00" >>job.slurm
  # Request cpus
  echo "#SBATCH --cpus-per-task ${n_cpus}" >>job.slurm
  # Specify RAM for operation
  echo "#SBATCH ---mem-per-cpu ${mem}" >>job.slurm
  # Specify partition (gpu)
  echo "#SBATCH ---partition ${device}" >>job.slurm
  # Specify CUDA GPU
  echo "#SBATCH ---gres ${cuda_specs}" >>job.slurm
  # Write output log to log directory
  echo "#SBATCH --output ${log_dir}/${method}/reward_networks-${method}_%j_$(date '+%Y_%m-%d_%H-%M-%S').out" >>job.slurm
  # Start in current directory:
  echo "#SBATCH --workdir ${code_dir}" >>job.slurm
  # Activate virtual environment on cluster
  echo "source .venv/bin/activate" >>job.slurm
  # Call main python script
  echo "python3 ${main_script_fn} ${method}" >>job.slurm
  echo "python3 ${main_script_fn} ${method}"
  # Submit job to cluster and remove job
  echo "REWARD NETWORKS ${method} in queue"
  echo ""

  sbatch job.slurm
  rm -f job.slurm

fi

