import torch
from model import DQN
from logger import AgentLogger

class ClusteredAgent:

    def __init__(self, cluster_kw, dqn_kw):
        self.cluster_kw = cluster_kw
        self.dqn_kw = dqn_kw
        self.agents = []
        self.n_clusters = -1

    def assign_cluster(self, state):
        if self.cluster_kw.get("force_one", False):
            return 0
        if state.sum() < 12:
            return 0
        return 1

    def init_clusters(self, states):
        self.n_clusters = 2
        if self.cluster_kw.get("force_one", False):
            self.n_clusters = 1
        for i in range(self.n_clusters):
            agent = DQN(**self.dqn_kw)
            self.agents.append(agent)
        AgentLogger("clusters:", self.n_clusters)

    def train_warmup(self, states, targets):
        self.init_clusters(states)
        clusters = []
        for i in range(states.size(0)):
            c = self.assign_cluster(states[i])
            clusters.append(c)
        clusters = torch.tensor(clusters).long()
        loss = []
        for n in range(self.n_clusters):
            mask = clusters == n
            l = self.agents[n].train_warmup(states[mask], targets[mask])
            loss.append(l)
        return loss

    def choose_actions(self, states):
        clusters = []
        for i in range(states.size(0)):
            c = self.assign_cluster(states[i])
            clusters.append(c)
        clusters = torch.tensor(clusters).long()
        randoms = torch.zeros(states.size(0)).bool()
        actions = torch.zeros(states.size(0)).long()
        for n in range(self.n_clusters):
            mask = clusters == n
            random, act = self.agents[n].choose_actions(states[mask])
            randoms[mask] = random
            actions[mask] = act
        return randoms, actions, clusters

    def weekly_training_update(self, transitions, index):
        AgentLogger("clustering transitions")
        clustered = [[] for x in range(self.n_clusters)]
        for t in transitions:
            c = self.assign_cluster(t[0])
            clustered[c].append(t)
        loss = []
        for n in range(self.n_clusters):
            dqn = self.agents[n]
            clust = clustered[n]
            l = dqn.weekly_training_update(clust, index)
            loss.append(l)
        return loss

    def save_disk_repr(self, stor, index):
        cdisk = {}
        cdisk['n_clusters'] = torch.tensor(self.n_clusters)
        AgentLogger("Saving agents", btbrk=None)
        for n in range(self.n_clusters):
            disk = {}
            dqn = self.agents[n]
            disk['policy_state_dict'] = dqn.policy.state_dict()
            disk['target_state_dict'] = dqn.target.state_dict()
            disk['memory'] = dqn.memory.mem
            disk['epsilon'] = torch.tensor(dqn.epsilon)
            cdisk[str(n)] = disk
            AgentLogger("...", n, topbrk=None, btbrk=None)
        AgentLogger("saved all", topbrk=None)
        return stor.save_data(cdisk, index)

    def load_disk_repr(self, stor, index):
        cdisk = stor.load_indexed_data(index)
        AgentLogger("Loading agent", btbrk=None)
        if cdisk is None:
            AgentLogger("initializing agent", topbrk=None)
        else:
            self.n_clusters = cdisk['n_clusters'].item()
            for i in range(self.n_clusters):
                agent = DQN(**self.dqn_kw)
                self.agents.append(agent)
            for n in range(self.n_clusters):
                AgentLogger("...", n, topbrk=None, btbrk=None)
                disk = cdisk[str(n)]
                self.agents[n].policy.load_state_dict(disk['policy_state_dict'])
                self.agents[n].target.load_state_dict(disk['target_state_dict'])
                self.agents[n].memory.mem = disk['memory']
                self.agents[n].epsilon = disk['epsilon'].item()
            AgentLogger("loaded all", topbrk=None)

if __name__ == "__main__":
    c = ClusteredAgent({})
    states = torch.randn(5, 8)
    targets = torch.randn(5, 1596)
    c.init_clusters(states)
    print(c.agents)
    w = c.train_warmup(states, targets)
    r, a = c.choose_actions(states)
    print(r)
    print(a)
    disk = c.save_disk_repr(None, 1)
    print(disk['1'])
    c2 = ClusteredAgent({})
    c2.load_disk_repr(disk, None, 1)
    disk2 = c2.save_disk_repr(None, 1)
    print(disk['1']['policy_state_dict']['net.0.weight'] == disk2['1']['policy_state_dict']['net.0.weight'])
