import torch
from sklearn.cluster import KMeans
from model import DQN
from logger import AgentLogger

class ClusteredAgent:

    def __init__(self, cluster_kw, dqn_kw):
        self.cluster_kw = cluster_kw
        self.dqn_kw = dqn_kw
        self.agents = []
        self.n_clusters = -1
        self.cluster_centers = None
    
    def init_clusters(self, states):
        clust = KMeans().fit(states)
        centers = torch.tensor(clust.cluster_centers_)
        self.n_clusters = centers.size(0)
        self.cluster_centers = centers
        for i in range(self.n_clusters):
            agent = DQN(**self.dqn_kw)
            self.agents.append(agent)
        AgentLogger("clusters:", self.n_clusters)

    def assign_cluster(self, state):
        d = (self.cluster_centers - state).pow(2).sum(1)
        return d.argmin().item()

    def assign_clusters(self, states):
        clusters = []
        for i in range(states.size(0)):
            c = self.assign_cluster(states[i])
            clusters.append(c)
        return torch.tensor(clusters).long()

    def train_warmup(self, clusters, states, targets):
        loss = []
        for n in range(self.n_clusters):
            mask = clusters == n
            l = self.agents[n].train_warmup(states[mask], targets[mask])
            loss.append(l)
        return loss

    def choose_actions(self, clusters, states):
        if not isinstance(clusters, torch.Tensor):
            clusters = torch.tensor(clusters)
        clusters = clusters.long()
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
            clustered[t[0]].append(t[1:])
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
        cdisk['cluster_centers'] = self.cluster_centers
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
            self.cluster_centers = cdisk['cluster_centers']
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
