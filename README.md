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
TODO
```

## Repo organization
* **notebooks**: includes `.ipynb`,`.py`,`.sh` files to generate reward networks and solve them
* **models**: includes `.py` files used for parsing and validation of JSON data
* **params**: includes `.yml` files for each file in notebooks specifying parameters
* **rn**: includes utilities used in the scripts in `notebooks` folder
* **data**: includes json files where all generated reward networks used in the experiment are stored. The `_viz` suffix in the file names indicate those data files that have additional node location information (for frontend vizualization purposes)


## Workflow
### Network generation
```mermaid

flowchart TD

subgraph Generation
A(params/generation.yml) --> B(notebooks/generation.ipynb)
W(models/network.py) --> B(notebooks/generation.ipynb)
H(notebooks/models.py) --> B(notebooks/generation.ipynb)
G(notebooks/utils.py) --> B(notebooks/generation.ipynb)
B(notebooks/generation.ipynb) --> C(data/train.json)
B(notebooks/generation.ipynb) --> X(data/test.json)
end
```

### Rule-based strategy comparisons
```mermaid

flowchart TD

subgraph Rule-based
A(params/environment.yml) --> L(notebooks/environment.py)
L(notebooks/environment.py) --> B(notebooks/rule_based.ipynb)
C(data/train.json) --> B(notebooks/rule_based.ipynb)
C(data/train.json) --> D(notebooks/try_vectorization.ipynb)
end
```

### DQN 
```mermaid

flowchart TD

subgraph DQN
A(params/dqn_agent.yml) --> B(notebooks/dqn_agent.py)
D(notebooks/environment_vect.py) --> B(notebooks/dqn_agent.py)
C(data/train.json) --> B(notebooks/dqn_agent.py) 
end
```



