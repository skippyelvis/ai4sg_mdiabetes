import torch
from torch import nn, optim
import numpy as np
from logger import DQNLogger

DEVICE = torch.device('cpu')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DQNLogger("device:", str(DEVICE))

class Memory:
    # memory object for holding most recent transitions

    def __init__(self, capacity=500, batch_size=150, **kw):
        self.capacity = capacity
        self.batch_size = batch_size
        self.keys = ['state', 'action', 'reward', 'next_state']
        self.mem = {}
        self.repl = 0

    @property
    def N(self):
        vec = self.mem.get(self.keys[0])
        return 0 if vec is None else vec.size(0)

    def add(self, transitions):
        for tr in transitions:
            repl = self.N >= self.capacity
            self._add(tr, repl)
            if repl:
                self.repl = (self.repl + 1) % self.capacity

    def _add(self, transition, repl):
        for i, k in enumerate(self.keys):
            vec = self.mem.get(k)
            telem = transition[i].reshape(1,-1)
            if vec is None:
                vec = telem
            else:
                if not repl:
                    vec = torch.cat((vec, telem))
                else:
                    vec[self.repl] = telem
            vec = vec.clone()
            self.mem[k] = vec

    def sample(self, device=None):
        if self.N == 0:
            return None
        batch_size = min(self.batch_size, self.N)
        idxs = np.random.choice(self.N, batch_size, replace=False)
        idxs = torch.tensor(idxs).long()
        samp = {}
        for key in self.keys:
            vec = self.mem[key][idxs].detach().clone()
            if device is not None:
                vec = vec.to(device)
            samp[key] = vec 
        return samp

class Model(nn.Module):
    # simple linear model as basis for dqn

    def __init__(self, input_shape=8, hidden_shape=128, output_shape=1596, hidden_layers=2,
            hidden_activation='LeakyReLU', hidden_activation_kw={'negative_slope': 0.2}):
        super().__init__()
        self.dev = DEVICE 
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        self.hidden_activation_kw = hidden_activation_kw
        self.net = self.build()

    def desc(self):
        out = {}
        out['input_shape'] = self.input_shape
        out['hidden_shape'] = self.hidden_shape
        out['output_shape'] = self.output_shape
        out['hidden_layers'] = self.hidden_layers
        out['hidden_activation'] = self.hidden_activation
        out['hidden_activation_kw'] = self.hidden_activation_kw
        return out

    def _pre(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = x.to(self.dev)
        return x

    def forward(self, x):
        x = self._pre(x)
        return self.net(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def build(self):
        activation = getattr(nn, self.hidden_activation)(**self.hidden_activation_kw)
        net = []
        net.append(nn.Linear(self.input_shape, self.hidden_shape))
        net.append(activation)
        for h in range(self.hidden_layers):
            net.append(nn.Linear(self.hidden_shape, self.hidden_shape))
            net.append(activation)
        net.append(nn.Linear(self.hidden_shape, self.output_shape))
        return nn.Sequential(*net)
    
    def clamp_grads(self):
        for param in self.parameters():
            param.grad.data.clamp_(-1,1)

class ConvergenceCheck:
    # helper to check if we have converged

    def __init__(self, reps_min_loss=-1, min_loss=-1, reps_min_delta=-1, min_delta=-1):
        self.reps_min_loss = reps_min_loss
        self.min_loss = min_loss
        self.reps_min_delta = reps_min_delta
        self.min_delta = min_delta
        self.c = 0
        self.c_min_delta = 0
        self.prev_loss = None

    def __call__(self, l):
        if self.prev_loss is not None:
            delt = (l-self.prev_loss).abs() / self.prev_loss
            if delt <= self.min_delta:
                self.c_min_delta += 1
            else:
                self.c_min_delta = 0
        if l < self.min_loss or torch.isnan(l):
            self.c += 1
        else:
            self.c = 0
        self.prev_loss = l.clone()
        if self.c >= self.reps_min_loss:
            return 1
        if self.c_min_delta >= self.reps_min_delta:
            return 2
        return None 

class DQN:
    # dqn agent

    def __init__(self, criterion='MSELoss', optimizer='SGD', optimizer_kw={}, train_lr=1e-7, 
            warmup_lr=1e-1, warmup_iters=100000, num_samples=200, 
            sync_steps=2, epsilon=0.9, epsilon_decay=0.9, gamma=.05, model={}, memory={}, convergence={}):
        self.dev = DEVICE
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_kw = optimizer_kw
        self.train_lr = train_lr
        self.warmup_lr = warmup_lr
        self.warmup_iters = warmup_iters
        self.num_samples = num_samples
        self.sync_steps = sync_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.policy = Model(**model).to(self.dev)
        self.target = Model(**model).to(self.dev)
        self.memory = Memory(**memory)
        self.convergence = convergence

    def check(self, train_or_warmup):
        cv = self.convergence[train_or_warmup]
        return ConvergenceCheck(**cv)

    def weekly_training_update(self, transitions, run_index):
        self.memory.add(transitions)
        lossh = torch.tensor([])
        if self.memory.N == 0:
            return lossh
        check = self.check("training")
        DQNLogger("Starting training", btbrk=None)
        optimizer = self.new_optimizer(self.train_lr)
        for ns in range(self.num_samples):
            sample_lossh = self.train_on_sample(optimizer)
            if sample_lossh is None:
                break
            lossh = torch.cat((lossh, sample_lossh))
            if ns % 25 == 0:
                lv = sample_lossh[-1].item()
                DQNLogger(" #", ns, f"/{self.num_samples}, Loss:", lv, topbrk=None, btbrk=None)
            ch = check(sample_lossh[-1])
            if ch == 1:
                DQNLogger("training minimum loss hit @", ns, topbrk=None, btbrk=None)
                break
            if ch == 2:
                DQNLogger("training minimum delta hit @", ns, topbrk=None, btbrk=None)
                break
        self.epsilon = self.epsilon * self.epsilon_decay
        self.sync_net_weights(run_index)
        DQNLogger("Done training", topbrk=None)
        return lossh

    def train_on_sample(self, optimizer):
        sample = self.memory.sample(self.dev)
        if sample is None:
            return None
        lossh = []
        criterion = self.new_criterion()
        optimizer.zero_grad()
        policy_qvals = self.policy(sample['state']).gather(1, sample['action'])
        target_reward = sample['reward']
        future = self.target.predict(sample['next_state'])
        future = self.gamma * future.max(1)[0].reshape(-1,1)
        target_reward += future
        loss = criterion(policy_qvals, target_reward)
        loss.backward()
        self.policy.clamp_grads()
        optimizer.step()
        lossh.append(loss.item())
        return torch.tensor(lossh) 

    def train_warmup(self, warmup_states, warmup_targets):
        warmup_targets = warmup_targets.to(self.dev)
        lossh = []
        optimizer = self.new_optimizer(self.warmup_lr)
        criterion = self.new_criterion()
        check = self.check("warmup")
        DQNLogger("Starting warmup", btbrk=None)
        for warmi in range(self.warmup_iters):
            pred_qvals = self.policy(warmup_states)
            loss = criterion(pred_qvals, warmup_targets)
            optimizer.zero_grad()
            loss.backward()
            self.policy.clamp_grads()
            optimizer.step()
            lossh.append(loss.item())
            if warmi % 25 == 0:
                DQNLogger(" #", warmi, ", Loss:", loss.item(), topbrk=None, btbrk=None)
            ch = check(loss)
            if ch == 1:
                DQNLogger("Warmup minimum loss hit @", warmi, topbrk=None, btbrk=None)
                break
            elif ch == 2:
                DQNLogger("Warmup minimum delta hit @", warmi, topbrk=None, btbrk=None)
                break
        self.sync_net_weights(0)
        del optimizer
        DQNLogger("Done warmup", topbrk=None)
        return torch.tensor(lossh)

    def choose_actions(self, states):
        randoms = torch.zeros(states.size(0)).bool()
        preds = torch.zeros(states.size(0)).long()
        for row in range(states.size(0)):
            trand = torch.rand(1)
            if trand < self.epsilon:
                pred = torch.randn(1, self.policy.output_shape)
                randoms[row] = True
            else:
                state = states[[row]]
                pred = self.policy.predict(state)
                randoms[row] = False
            preds[row] = pred.detach().cpu().argmax(1)
        return (randoms, preds)

    def sync_net_weights(self, idx):
        # possibly update target net weights
        if idx % self.sync_steps != 0:
            return 
        DQNLogger("syncing network weights", topbrk=None, btbrk=None)
        self.target.load_state_dict(self.policy.state_dict())

    def new_optimizer(self, lr):
        # helper function to find and load new optimizer
        opt = getattr(optim, self.optimizer)
        return opt(self.policy.parameters(), lr=lr, **self.optimizer_kw)

    def new_criterion(self):
        # helper to get torch criterion function
        crit = getattr(nn, self.criterion)()
        return crit.to(self.dev)

    def save_disk_repr(self, stor, index):
        # dump a record of this dqn agent to disk
        disk = {}
        disk['policy_state_dict'] = self.policy.state_dict()
        disk['target_state_dict'] = self.target.state_dict()
        disk['memory'] = self.memory.mem
        disk['epsilon'] = torch.tensor(self.epsilon)
        DQNLogger("saving DQN#", index)
        return stor.save_data(disk, index)

    def load_disk_repr(self, stor, index):
        # load the most recent dqn agent
        disk = stor.load_indexed_data(index)
        DQNLogger("loading DQN#", index, btbrk=None)
        if disk is not None:
            self.policy.load_state_dict(disk['policy_state_dict'])
            self.target.load_state_dict(disk['target_state_dict'])
            self.memory.mem = disk['memory']
            self.epsilon = disk['epsilon'].item()
            DQNLogger("loaded from gcs", topbrk=None)
        else:
            DQNLogger("using randomly initialized models", topbrk=None)

