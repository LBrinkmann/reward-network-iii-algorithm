""" script for calculating probabilities of ending up in a destination node in any step with
 a random walk in the environments.
"""

import random
import pandas as pd
import numpy as np
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
pd.options.mode.chained_assignment = None

lst_col = 'actions'


def parse_rewards(row):
    return row['actions']['reward']


def parse_sourceId(row):
    return row['actions']['sourceId']


def parse_targetId(row):
    return row['actions']['targetId']


def parse_step(row):
    return row['actions']['step']


def parse_takebest(row):
    return row['checkTakebest'][row['step']-1]


def is_takebest_random_walk(row):
    ''' check if action is taken with Takebest strategy'''
    istakebest = []
    for i in range(len(row['nodeTrace'])-1):
        takebest = row['rewardMatrix'][row['nodeTrace'][i]].max()
        if row['rewardMatrix'][row['nodeTrace'][i], row['nodeTrace'][i+1]] < takebest:
            istakebest.append(0)
        else:
            istakebest.append(1)
    return istakebest


def random_walk(row):
    ''' random walk in envrionments'''
    NT = []
    RT = []
    for i in range(8):
        if i == 0:
            NT.append(row['startingNode'])
            reward = random.choice(row['rewardMatrix'][row['startingNode']]
                                   [np.nonzero(row['rewardMatrix'][row['startingNode']])])
            target = random.choice(np.where(row['rewardMatrix'][row['startingNode']] == reward)[0])
            NT.append(target)
            RT.append(reward)
        else:
            reward = random.choice(row['rewardMatrix'][target]
                                   [np.nonzero(row['rewardMatrix'][target])])
            target = random.choice(np.where(row['rewardMatrix'][target] == reward)[0])
            NT.append(target)
            RT.append(reward)
    return pd.Series([NT, RT])


def add_actions(row):
    actions_list = []
    for i in range(len(row['rewardTrace'])):
        actions_list.append({'reward': row['rewardTrace'][i], 'sourceId': row['nodeTrace']
                             [i], 'targetId': row['nodeTrace'][i+1], 'step': i+1})
    return actions_list


def expand_actions(df_walk, lst_col):
    '''expand lst_col (actions) column to rows. 1 row -> 8 rows '''
    df_walk_actions = pd.DataFrame({col: np.repeat(df_walk[col].values, df_walk[lst_col].str.len())
                                    for col in df_walk.columns.difference([lst_col])}).assign(**{lst_col: np.concatenate(df_walk[lst_col].values)})[df_walk.columns.tolist()]
    return df_walk_actions


def main(df_total, sample_size):
    df_walk = df_total[['environmentId', 'rewardMatrix', 'sourceId']].copy()
    df_walk['type'] = 'randomWalk'
    df_walk['startingNode'] = df_walk['sourceId']
    df_walk.pop('sourceId')
    df_walk.drop_duplicates(subset='environmentId', inplace=True, ignore_index=True)
    df_walk = df_walk.append([df_walk]*(sample_size-1), ignore_index=True)
    a = df_walk.apply(random_walk, axis=1)
    df_walk['nodeTrace'] = a.iloc[:, 0].copy()
    df_walk['rewardTrace'] = a.iloc[:, 1].copy()
    df_walk['checkTakebest'] = df_walk.apply(is_takebest_random_walk, axis=1)
    df_walk['actions'] = df_walk.apply(add_actions, axis=1)
    df_walk_actions = expand_actions(df_walk, lst_col)
    df_walk_actions['step'] = df_walk_actions.apply(parse_step, axis=1)
    df_walk_actions['actionReward'] = df_walk_actions.apply(parse_rewards, axis=1)
    df_walk_actions['sourceId'] = df_walk_actions.apply(parse_sourceId, axis=1)
    df_walk_actions['targetId'] = df_walk_actions.apply(parse_targetId, axis=1)
    df_walk_actions['isTakebest'] = df_walk_actions.apply(parse_takebest, axis=1)
    df_prob = df_walk_actions.groupby(
        ['environmentId', 'sourceId', 'targetId', 'step']).count()/sample_size
    return df_prob
