# Reward Network II

## Install required packages

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

## Run

```

```

## Repo organization
* **notebooks**: includes `.ipynb`,`.py`,`.sh` files to generate reward networks and solve them
* **models**: includes `.py` files used for parsing and validation of JSON data
* **params**: includes `.yml` files for each file in notebooks specifying parameters
* **rn**: includes utilities used in the scripts in `notebooks` folder

## Workflow
```mermaid

flowchart TD

subgraph Generation
A(params/generation.yml) --> B(notebooks/generation.ipynb)
B(notebooks/generation.ipynb) --> C(train.json)
B(notebooks/generation.ipynb) --> X(test.json)
W(models/network.py) --> C(train.json)
W(models/network.py) --> X(test.json)
end

subgraph Rule_based_Agents
D(params/environment.yml) --> E(notebooks/environment.py) 
E(notebooks/environment.py) --> F(notebooks/rule_based.ipynb)
end

subgraph DQN
G(params/dqn.yml) --> H(notebooks/dqn.py)
H(notebooks/dqn.py) --> I(notebooks/dqn_submit_job.sh)
end


Generation --> Rule_based_Agents
Generation --> DQN

```


