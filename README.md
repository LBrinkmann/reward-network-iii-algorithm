# Reward Network III

## Install required packages

Here Python 3.10.8 was used

```bash
python3 -m venv .venv
# Mac/Linux
. .venv/bin/activate 
# Windows
# source .venv/Scripts/Activate 


pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
```

## Repo organization
* **rniii** is the project folder where all python scripts are found, in particular inside this folder we find
  * **generate**: includes `.py` scripts to generate reward networks, as well as pydantic models to validate the generated networks structure
  * **solve**: includes `.py` scripts to solve networks according to rule-based strategies
  * **dqn**: includes `.py` files used for solving networks through deep reinforcement learning, as well as `.sh`files used to submit dqn runs on cluster
* **params**: includes `.yml` files relevant to each subfolder (generate, solve, dqn)
* **rn**: includes utilities used in the scripts in `notebooks` folder
* **data**: includes json files where all generated reward networks used in the experiment are stored. The `_viz` suffix in the file names indicate those data files that have additional node location information (for frontend vizualization purposes)

## Workflow

### Network generation

```mermaid

flowchart TD

subgraph Generation
A(params/generate/generation.yml) --> B(generate/generation.py)
W(generate/network.py) --> B(generate/generation.py)
H(generate/environment.py) --> B(generate/generation.py)
B(generate/generation.py) --> C(data/networks.json)
end
```

### Rule-based strategy comparisons

```mermaid

flowchart TD

subgraph Rule-based
A(params/rule_based_solve/environment.yml) --> L(solve/rule_based.py)
G(solve/environment.py) --> L(solve/rule_based.py)
L(solve/rule_based.py) --> B(data/solutions.json)
L(solve/rule_based.py) --> D(data/solutions.csv)
end
```

### DQN + Wandb

Training the DQN model both locally and on cluster is organized in the `wandb_on_slurm.py` script.

First, we create a sweep ID (command `create`) by specifying a parameter grid file: this file contains the path to the script to run (in our case `notebooks/dqn_agent.py`), parameter names and respective values.

```python
python wandb_on_slurm.py create params/dqn/grid.yaml
```

After we get the sweep ID we can:

* run the sweep locally (command `local` followed by sweep ID)

```python
python wandb_on_slurm.py local <sweep_id>
```

* run the sweep on SLURM cluster (command `slurm` followed by sweep ID and the worker in the cluster)

```python
python wandb_on_slurm.py slurm <sweep_id> <number or worker>
```

```mermaid

flowchart TD

subgraph SweepID
A(params/dqn/grid.yaml) --> B(wandb_on_slurm.py)
B(wandb_on_slurm.py) --> C(sweep ID)
end

C(sweep ID) --> D{Local or SLURM cluster?}
D{Local or SLURM cluster?} --> E(wandb_on_slurm.py) & G(wandb_on_slurm.py)

subgraph Local
E(wandb_on_slurm.py) --> F(dqn/dqn_agent.py)
end

subgraph Slurm
H(dqn/dqn_agent.py) --> G(wandb_on_slurm.py)
I(dqn/slurm_template.sh) --> G(wandb_on_slurm.py)
end
```

The script `dqn_agent.py` solves environments all at once using both DQN and rule based methods: the rule based methods results are also logged and serve as reference in the metrics plots.

We log metrics for each episode: metrics for rule based and for dqn can be then compared in wandb following instructions at https://docs.wandb.ai/ref/app/features/panels/line-plot .