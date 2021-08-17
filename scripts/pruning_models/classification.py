import pandas as pd
from scripts.pruning_models.model import eval_network


def _eval_network(kwargs):
    return eval_network(**kwargs)


# used by: preselection.py,
def apply_models(data, model_settings, n_steps, n_nodes=None):
    """
    Creating df for different models.
    """
    if n_nodes is None:
        n_nodes = len(data[0]['nodes'])
    network_list = []
    for network in data:
        for model_setting in model_settings:
            print('apply_models')
            print(network)
            network_list.extend(
                eval_network(network, n_nodes=n_nodes, n_steps=n_steps, **model_setting)
            )
    df = pd.DataFrame(data=network_list)
    return df
