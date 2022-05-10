import torch
from content import MessageHandler

MessagesH = MessageHandler()

# calculate the entropy of messages sent
def entropy(x):
    x = torch.cat((x, torch.tensor([0]), torch.tensor([1596])))
    b = x.bincount().float()
    b[0] -= 1
    b[-1] -= 1
    b = torch.nn.functional.softmax(b, dim=0)
    b = torch.distributions.Categorical(b).entropy()
    return b

# calculate the scores of actions based on states
def scoreai(states, actions):
    scores = torch.full_like(actions, -1.).float()
    weak_sids = states.argsort()
    sids = torch.tensor([MessagesH.sid_lookup(a) for a in actions]).long()
    for row in range(actions.size(0)):
        score = 0
        for sid in sids[row]:
            sid -= 1
            sid_idx = (sid == weak_sids[row]).nonzero()[0]
            score += sid_idx 
        scores[row] = score/len(sids[row])
    return scores

def debugai(mask, actions, ai_random, loss, ids, clusters, states, init_cluster_debug, cluster_t_counts):
    actions = actions[:,1]
    ids = ids[mask]
    clusters = clusters[mask]
    states = states[mask].clone()
    actions = actions[mask]
    debug = {
            "friendly": torch.tensor([]),
            "metrics": {},
            "loss": loss,
            "friendly_keys": ["id", "score", "action", "random", "cluster"]
    }
    if actions.size(0) == 0:
        return debug 
    scores = scoreai(states, actions)
    debug["friendly"] = torch.cat((
        ids.reshape(-1,1),
        scores.reshape(-1,1), 
        actions.reshape(-1,1),
        ai_random.reshape(-1,1),
        clusters.reshape(-1,1)),1)
    debug["metrics"] = {
        "average_ai_score": scores[~ai_random].float().mean(),
        "n_unique": actions.unique().size(0),
        "action_representation": actions.unique().size(0) / MessagesH.N,
        "action_entropy": entropy(actions),
        "cluster_t_counts": cluster_t_counts,
    }
    if init_cluster_debug is not None:
        debug["init_cluster_metrics"] = init_cluster_debug
    return debug
