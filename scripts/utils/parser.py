import hashlib


def parse_node(name_map,pos_map,id):
    return {
        'id': id-1,
        'displayName': name_map[id-1],
        **pos_map[id-1]
    }


def parse_link(reward_map,source, target, weight, **_):
    return {
        "reward": weight,
        "rewardName": reward_map[weight],
        "sourceId": source - 1,
        "targetId": target - 1
    }


def parse_network(
        model_scores, experiment_name, *, nodes, links, graph, network_id,
        environment_id, startingNodeId, Class, **kwargs):
    return {
        'type': 'network',
        'version': 'four-rewards-v2',
        '_id': hashlib.md5((environment_id + experiment_name).encode()).hexdigest(),
        'experimentEnvironmentId': hashlib.md5((environment_id + experiment_name).encode()).hexdigest(),
        'environmentId': environment_id,
        'startingNodeId': startingNodeId,
        'networkId': network_id,
        'requiredSolutionLength': 8,
        'missingSolutionPenalty': -500,
        'baselineReward': model_scores.loc[environment_id, 'TakeBest'],
        'class': Class,
        'experimentName': experiment_name,
        'actions': [parse_link(**l) for l in links],
        'nodes': [parse_node(**n) for n in nodes]
    }

def parse_network_v2(requiredSolutionLength,missingSolutionPenalty,experiment_name,reward_map,name_map,pos_map,*, nodes, links, graph, network_id,
        environment_id, startingNodeId, Class, **kwargs):
    return {
        'type': 'network',
        'version': 'four-rewards-v2',
        '_id': hashlib.md5((environment_id + experiment_name).encode()).hexdigest(),
        'experimentEnvironmentId': hashlib.md5((environment_id + experiment_name).encode()).hexdigest(),
        'environmentId': environment_id,
        'startingNodeId': startingNodeId,
        'networkId': network_id,
        'requiredSolutionLength': requiredSolutionLength,
        'missingSolutionPenalty': missingSolutionPenalty,
        'class': Class,
        'experimentName': experiment_name,
        'actions': [parse_link(reward_map,**l) for l in links],
        'nodes': [parse_node(name_map,pos_map,**n) for n in nodes]
    }
def parse_actions(node_trace, reward_trace):
    return [
        {**parse_link(s+1, t+1, r), 'step': i + 1}
        for i, (s, t, r) in enumerate(zip(node_trace[:-1], node_trace[1:], reward_trace))
    ]


def parse_solution(
        Solution_ID, Environment_ID, Total_Reward, Node_Trace, Reward_Trace,
        Model_Name, **_):
    return {
        "actions": parse_actions(Node_Trace, Reward_Trace),
        "environmentId": Environment_ID,
        "solutionId": Solution_ID,
        "modelName": Model_Name,
        "totalReward": Total_Reward
    }
