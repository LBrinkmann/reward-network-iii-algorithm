"""Usage: preselection.py PARAMETER_YAML NETWORKS_JSON OUTPUT_FOLDER

An environment is defined by the tuple networkId and startingNode.
This script allows preselection of environments based on different metrics.

Metrics:
    Lookahead Performance:          Maximum reward of a environment.
    TakeBest Performance:           Expected reward of a greedy algorithm without planing.
    Lookahead Risky Actions:        Number of risky (-100) actions of the lookahead solution.
    Lookahead Non Trivial Actions:  Number of actions of the lookahead solution, where the
                                    counterfactual action has higher reward.
    Lookahead Unique Nodes:         Number of unique node the lookahead solution covers.
    Max Equal Trace:                A similarity metric between two solutions.

Arguments:
    PARAMETER_YAML        A yaml parameter file.
    NETWORKS_JSON         A json file with networks.
    OUTPUT_FOLDER         A folder used for the outputs.

Outputs:
    DataFrame:
        Network_ID:     NetworkId
        Starting_Node:  Starting node of environment.
        Preselection:   Name of the Preselection
"""


from docopt import docopt
import pandas as pd
from scripts.utils import store
from scripts.utils.utils import load_yaml
from scripts.pruning_models.classification import apply_models
import ipdb


def LCSubStr(X, Y, m, n):
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]
    result = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result


def reward_features(df):
    """
    Metrics based on reward.
    """
    return (
        df.loc['Lookahead', 'Total_Reward'].rename("Lookahead Performance").to_frame(),
        (
            df.loc['Lookahead', 'Total_Reward'] - df.loc['TakeBest', 'Total_Reward']
        ).rename("Lookahead TakeBest Performance Difference").to_frame(),
    )


def node_trace(df):
    """
    Metrics based the node trace of solutions.
    """
    return (
        df.loc['Lookahead', 'Node_Trace'].apply(lambda x: len(
            set(x))).rename("Lookahead Unique Nodes").to_frame(),
        (
            df['Node_Trace'].unstack('Model_Name')
            .apply(lambda sr: LCSubStr(sr['TakeBest'], sr['Lookahead'], 9, 9), axis=1) - 1
        ).rename("Lookahead TakeBest Max Equal Trace").to_frame()
    )


def complexity(df):
    """
    Metrics based on counterfactual reward difference between actions from the same node.
    """
    return (
        df.loc['Lookahead'].apply(diff_trace_metrics, axis=1),
    )


def diff_trace_metrics(row):
    n_non_trivial_decisions = sum(
        1 for rd, qd in zip(row["Reward_Diff"], row['Q_Diff']) if ((qd > 0) & (rd <= 0)))
    n_risky_decisions = sum(
        1 for rd, qd in zip(row["Reward_Diff"], row['Q_Diff']) if ((qd > 0) & (rd < 0)))
    return pd.Series({
        "Lookahead Risky Actions": n_risky_decisions,
        "Lookahead Non Trivial Actions": n_non_trivial_decisions
    })


def preselection_outlier(df):
    """
    Select enviroments with their lookahead performance being in the central quantile.
    """
    return df[(df["Lookahead Performance"] >= df["Lookahead Performance"].quantile(.25)) &
              (df["Lookahead Performance"] <= df["Lookahead Performance"].quantile(.75))]


def preselection_query(df, query):
    return df.query(query)


def preselection_step(df, ptype, **kwargs):
    if ptype == 'outlier':
        return preselection_outlier(df, **kwargs)
    if ptype == 'query':
        return preselection_query(df, **kwargs)


def one_preselection(df, name, steps):
    for s in steps:
        df = preselection_step(df, **s)
    df['Preselection'] = name
    return df


def all_preselection(df, preselections):
    return pd.concat(
        [one_preselection(df, **p).copy() for p in preselections],
        axis=0
    ).reset_index(drop=False)


def parse_node(name_map, pos_map, id):
    return {
        'id': id,
        'displayName': name_map[id],
        **pos_map[id]
    }


def parse_link(reward_map, source, target, weight, **_):
    return {
        "reward": weight,
        # "rewardName": reward_map[weight],
        "sourceId": source,
        "targetId": target
    }


def parse_network_v2(requiredSolutionLength, missingSolutionPenalty, experiment_name, reward_map, name_map, pos_map, *, nodes, links, graph, network_id, **kwargs):
    return {
        'type': 'network',
        'version': 'four-rewards-v2',
        'network_id': network_id,
        'actions': [parse_link(reward_map, **l) for l in links],
        'nodes': [parse_node(name_map, pos_map, **n) for n in nodes]
    }


def process(data, preselections, **params):
    df = apply_models(data, **params)
    df = df.set_index(['Model_Name', 'Environment_ID', 'Network_ID', 'Starting_Node', 'Depth'])
    dfs = reward_features(df) + node_trace(df) + complexity(df)

    dfn = pd.concat(dfs, axis=1)
    dfp = all_preselection(dfn, preselections)

    return dfp[['Network_ID', 'Starting_Node', 'Preselection']]


def main(parameter_yaml, networks_json, output_folder):
    # modified
    # TODO: finalize modification
    params = load_yaml(parameter_yaml)
    data = store.load_json(networks_json)
    #data = {n['network_id']: n for n in data}
    networks_new = [parse_network_v2(
        **params['parse_network_v2'], **n) for n in data]
    df = process(networks_new, **params['rest'])
    df = df.reset_index()
    store.store_df(df, output_folder, 'preselection')
    store.store_json(networks_new, output_folder, 'networks')
    print('complete')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    arguments_low = {k.lower(): v for k, v in arguments.items()}
    main(**arguments_low)
