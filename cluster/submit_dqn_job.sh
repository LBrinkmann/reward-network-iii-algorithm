#!/bin/bash

# This script parallelizes the jobs related to the Reward Networks project on the MPIB Cluster (tardis)
# The process is executed on tardis using a job wrapper for SLURM resource management system.
# There are two possible "methods" in the script:
# - dqn: this trains the DQN agent but does NOT run hyperparameter tuning
# - tune: this trains the DQN agent AND runs hyperparameter tuning
#
# PROJECT: Reward Networks III
# author @ Sara Bonati
# Center for Humans and Machines - Max Planck Institute for Human Development - Berlin
#-----------------------------------------------------------------------------

# ---------------------------Worfklow settings--------------------------------
method="dqn" 

# -------Directory and filename (fn) management-------------------------------
home_dir="/mnt/beegfs/home/bonati"
project_dir="${home_dir}/CHM/reward_networks_III"
code_dir="${project_dir}/reward-network-iii-algorithm/notebooks"
data_dir="${project_dir}/data"
out_dir="${project_dir}/results"
log_dir="${project_dir}/logs"

# script path
main_script_fn=${code_dir}/notebooks/dqn_agent.py
# virtual environment path
venv_path="../.venv/bin/activate"


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


# --------Create SLURM job and submit to cluster----------------------

if [[ "${method}" == "dqn" ]]; then

  # Define computation parameters
  n_cpus=4 # maximum number of cpus per process
  mem=10GB # memory demand
  device="cpu"#"gpu"
  cuda_specs="gpu:pascal:1"

  # Create slurm job file
  echo "#!/bin/bash" >job.slurm
  # Name job file
  echo "#SBATCH --job-name reward_networks-${method}" >>job.slurm
  # Specify maximum run time
  echo "#SBATCH --time 10:00:00" >>job.slurm
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
  echo "python3 ${main_script_fn} ${data_dir} ${out_dir}" >>job.slurm
  echo "python3 ${main_script_fn} ${data_dir} ${out_dir}"
  # Submit job to cluster and remove slurm file
  echo "REWARD NETWORKS ${method} in queue"
  echo ""

  sbatch job.slurm
  rm -f job.slurm

fi


if [[ "${method}" == "tune" ]]; then
  
  # Define computation parameters
  n_cpus=4 # maximum number of cpus per process
  mem=10GB # memory demand
  device="gpu"
  cuda_specs="gpu:pascal:1"

  # Create slurm job file
  echo "#!/bin/bash" >job.slurm
  # Name job file
  echo "#SBATCH --job-name reward_networks-${method}" >>job.slurm
  # Specify maximum run time
  echo "#SBATCH --time 10:00:00" >>job.slurm
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
  echo "python3 ${main_script_fn} -t ${data_dir} ${out_dir}" >>job.slurm
  echo "python3 ${main_script_fn} -t ${data_dir} ${out_dir}"
  # Submit job to cluster and remove slurm file
  echo "REWARD NETWORKS ${method} in queue"
  echo ""

  sbatch job.slurm
  rm -f job.slurm

fi