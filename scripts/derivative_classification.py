"""Usage: derivative_classification.py PARAMETER_YAML NETWORKS_JSON PRESELECTIONS_DF OUTPUT_FOLDER

Classifies the environments based on the performance difference between two models. It assumes
that one model is risky and the other is risk aversive. Environments are classified as
'Human Regretfull', 'Neutral' and 'Riskseeking Regretfull'.

Arguments:
    PARAMETER_YAML        A yaml parameter file.
    NETWORKS_JSON         A json file with networks.
    PRESELECTIONS_DF      A dataframe with preselection.
    OUTPUT_FOLDER         A folder used for the outputs.

Outputs:
    DataFrame:
        Preselection:   Name of the Preselection.
        Network_ID:     NetworkId.
        Starting_Node:  Starting node of environment.
        Class:          Bin label of the classification.
"""


from docopt import docopt
import pandas as pd
from scripts.utils import store
from scripts.utils.utils import load_yaml
from scripts.pruning_models.calculate_regret import calc_regret


def classify_derivative(df, bins):
    diff = df['Risky'] - df['Aversive']
    diff_rank = diff.rank(pct=True)
    classes = pd.cut(
        diff_rank, bins, labels=['Human Regretfull', 'Neutral', 'Riskseeking Regretfull'],
        right=True
    ).astype(str)
    return classes


def process(networks, preselections, bins, model_settings, **params):
    settings_df = pd.DataFrame.from_records(model_settings)
    df = calc_regret(networks, settings_df, **params)

    df = df.set_index(['Model_Name', 'Network_ID', 'Starting_Node'])
    dfs = df['Regret'].unstack('Model_Name').reset_index()
    dfs = dfs.merge(preselections)

    dfs['Class'] = dfs.groupby('Preselection', group_keys=False).apply(
        classify_derivative, bins=bins)

    return dfs[['Preselection', 'Network_ID', 'Starting_Node', 'Class']]


def main(parameter_yaml, networks_json, preselections_df, output_folder):
    params = load_yaml(parameter_yaml)
    print('Parameter yaml loaded.')
    data = store.load_json(networks_json)
    print('Network json loaded.')
    preselections = store.load_df(preselections_df)
    print('Preselections df loaded.')
    df = process(data, preselections, **params)
    store.store_df(df, output_folder, 'classification')


if __name__ == "__main__":
    arguments = docopt(__doc__)
    arguments_low = {k.lower(): v for k, v in arguments.items()}
    main(**arguments_low)
