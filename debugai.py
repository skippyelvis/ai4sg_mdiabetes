import torch
from content import MessagesH

# calculate the entropy of messages sent
def entropy(x):
    x = torch.cat((x, torch.tensor([0]), torch.tensor([1596])))
    b = x.bincount().float()
    b[-1] -= 1
    b[-2] -= 1
    b = torch.nn.functional.softmax(b, dim=0)
    b = torch.distributions.Categorical(b).entropy()
    return b

# calculate the scores of actions based on states
def scoreai(states, actions):
    scores = torch.full_like(actions, -1.).float()
    weak_sids = states.argsort()
    sids = torch.tensor([MessagesH.sid_lookup(a) for a in actions]).long()
    for row in range(actions.size(0)):
        if ai_random[row]:
            continue
        score = 0
        for sid in sids[row]:
            sid -= 1
            sid_idx = (sid == weak_sids[row]).nonzero()[0]
            score += sid_idx 
        scores[row] = score/len(sids[row])
    return scores

def debugai(mask, actions, ai_random, loss, ids, clusters, states):
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
        "average_score": scores[~ai_random_mask].float().mean(),
        "n_unique": actions.unique().size(0),
        "action_representation": actions.unique().size(0) / MessagesH.N,
        "action_entropy": entropy(actions),
    }
    return debug
