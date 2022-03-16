import json
import os
import torch
from storage import make_storage_group, load_yaml
from storage import Storage
from agent import ClusteredAgent
from content import StatesH, MessagesH, QuestionsH
from logger import MainLogger

class MDiabetes:

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_yaml(self.config_path)
        self.dry_run = self.config["dry_run"]
        self.simulate_responses = self.config["simulate_responses"]
        self.simulate_participants = self.config["simulate_participants"]
        self.stor = make_storage_group(**self.config["storage"])
        self.agent = None 
        self.run_index = -5

    def main(self):
        self.run_index = self.stor["states"].count_files() + 1
        MainLogger("Starting week #", self.run_index)
        MainLogger("Dry run:", self.dry_run)
        MainLogger("Simulated Responses:", self.simulate_responses)
        MainLogger("Simulated Participants:", self.simulate_participants)
        MainLogger("Loading DQN agent")
        self.agent = ClusteredAgent(self.config["cluster"], self.config["dqn"])
        self.agent.load_disk_repr(self.stor["dqns"], self.run_index-1)
        if not self.simulate_participants:
            MainLogger("Gathering real participants")
            timeline, ids, clusters, states = self.gather_participants()
        else:
            MainLogger("Gathering simul participants")
            timeline, ids, clusters, states = self.gather_simulated_participants()
        MainLogger("# Participants:", states.size(0))
        weekly_loss = torch.tensor([])
        if self.run_index == 1:
            MainLogger("Warming up agent")
            weekly_loss = self.warmup_agent(clusters, states)
        optin, core, ai = self.weekly_masks(timeline)
        MainLogger("Generating weekly actions")
        actions, ai_random = self.weekly_actions(optin, core, ai, timeline, ids, clusters, states)
        MainLogger("Building message/question file")
        msg_qsn = self.generate_msg_qsn(actions, timeline, states)
        MainLogger("Checking for responses")
        prev_actions, prev_clusters, responses = self.collect_responses()
        MainLogger("updating states and adding transitions")
        next_states, transitions = self.update_states(prev_actions, prev_clusters, responses, states, ids)
        if self.run_index > 1 and len(transitions) > 0:
            weekly_loss = self.agent.weekly_training_update(transitions, self.run_index)
        MainLogger("analyzing ai performance")
        debug = self.weekly_ai_debug(ai, actions, ai_random, weekly_loss, ids, clusters, states)
        if not self.dry_run:
            MainLogger("Saving data")
            self.stor["states"].save_data(next_states, self.run_index)
            self.stor["ids"].save_data(ids, self.run_index)
            self.stor["clusters"].save_data(clusters, self.run_index)
            self.stor["actions"].save_data(actions, self.run_index)
            self.stor["timelines"].save_data(timeline, self.run_index)
            self.agent.save_disk_repr(self.stor["dqns"], self.run_index)
            self.stor["debugs"].save_data(debug, self.run_index)
            self.stor["outfiles"].save_data(msg_qsn, self.run_index)
            self.stor["yaml"].save_data(self.config_path, self.run_index)
        MainLogger("="*20)

    def gather_participants(self):
        # load previous timeline, ids, and states
        # load possible new batch
        timeline = self.stor["timelines"].load_indexed_data(self.run_index-1)
        ids = self.stor["ids"].load_indexed_data(self.run_index-1)
        clusters = self.stor["clusters"].load_indexed_data(self.run_index-1)
        states = self.stor["states"].load_indexed_data(self.run_index-1)
        new_batch = self.stor["batches"].load_indexed_data(self.run_index)
        def modify_whatsapp(x):
            x = str(x)
            x = x[len(x)-10:]
            return int(x)
        if new_batch is not None:
            all_whatsapps, all_states = StatesH.compute_states()
            new_states = None
            for new_glific_id, new_whatsapp in new_batch:
                new_whatsapp = modify_whatsapp(new_whatsapp)
                where = (new_whatsapp == all_whatsapps).nonzero()
                if where.size(0) == 0:
                    continue
                new_glific_id = torch.tensor([new_glific_id]).long()
                new_state = all_states[where[0]]
                new_tl = torch.cat((new_glific_id, torch.tensor([0]))).reshape(1,-1)
                if new_states is None:
                    new_states = new_state
                else:
                    new_states = torch.cat((new_states, new_state))
                if timeline is None:
                    timeline = new_tl
                    ids = new_glific_id
                else:
                    timeline = torch.cat((timeline, new_tl))
                    ids = torch.cat((ids, new_glific_id))
            if states is None:
                states = new_states
                MainLogger("Initializing clusters")
                self.init_agent_clusters(modify_whatsapp, all_whatsapps, all_states)
                clusters = self.agent.assign_clusters(states)
            else:
                states = torch.cat((states, new_states))
                new_clusters = self.agent.assign_clusters(new_states)
                clusters = torch.cat((clusters, new_clusters))
        timeline[:,1] += 1
        return timeline, ids, clusters, states 

    def init_agent_clusters(self, mwa, all_whatsapps, all_states):
        new_batch = Storage.load_csv("arogya_content/all_ai_participants.csv")
        idxs = []
        for gid, wa in new_batch:
            wa = mwa(wa)
            where = (wa == all_whatsapps).nonzero()
            if where.size(0) == 0:
                continue
            idxs.append(where)
        idxs = torch.tensor(idxs).long()
        self.agent.init_clusters(all_states[idxs])

    def gather_simulated_participants(self):
        timeline = self.stor["timelines"].load_indexed_data(self.run_index-1)
        ids = self.stor["ids"].load_indexed_data(self.run_index-1)
        states = self.stor["states"].load_indexed_data(self.run_index-1)
        N = 0
        if self.run_index == 1:
            N = self.config.get("N_simul_batch_1", 0)
        elif self.run_index == 2:
            N = self.config.get("N_simul_batch_2", 0)
        k = 0 if ids is None else ids.size(0)
        new_ids = torch.arange(N).long() + k
        low, high = 0, 3
        new_states = (low-high) * torch.rand(N,8) + high
        new_timeline = new_ids.reshape(-1,1)
        tl = torch.zeros_like(new_timeline)
        new_timeline = torch.cat((new_timeline, tl),1)
        if timeline is None:
            new_timeline[:,1] += 1
            return new_timeline, new_ids, new_states
        timeline = torch.cat((timeline, new_timeline))
        timeline[:,1] += 1
        ids = torch.cat((ids, new_ids))
        states = torch.cat((states, new_states))
        return timeline, ids, states

    def warmup_agent(self, clusters, states):
        targets = torch.zeros(states.size(0), MessagesH.N)
        mesage_sids = []
        for col in range(MessagesH.N):
            sids = MessagesH.sid_lookup(col)
            mesage_sids.append(sids)
        for row in range(states.size(0)):
            for col, sids in enumerate(mesage_sids):
                val = 0
                for sid in sids:
                    val += StatesH.state_max - states[row][sid-1]
                targets[row,col] = val ** (1/2)
        return self.agent.train_warmup(clusters, states, targets)

    def weekly_masks(self, timeline):
        # make the optin, core, ai groupings for this week
        optin_mask = torch.zeros(timeline[:,0].size()).bool()
        core_mask = optin_mask.clone() 
        ai_mask = optin_mask.clone() 
        for idx in range(timeline.size(0)):
            tl = timeline[idx,1]
            if tl == 1:
                optin_mask[idx] = True
            elif tl.long().item() in MessagesH.core_timeline_map:
                core_mask[idx] = True
            else:
                ai_mask[idx] = True
        return (optin_mask, core_mask, ai_mask)

    def weekly_actions(self, optin, core, ai, timeline, ids, clusters, states):
        # determine which actions to take this week
        # optin group gets random core action
        # core group needs scheduled action
        # ai group gets input to dqn
        actions = torch.zeros(states.size(0)).long()
        optin_actions = MessagesH.random_core_actions(optin.sum().item())
        core_actions = MessagesH.scheduled_core_actions(timeline[core])
        ai_random_mask, ai_actions, ai_clusters = self.agent.choose_actions(clusters[ai], states[ai])
        actions[optin] = torch.tensor(optin_actions).long() 
        actions[core] = torch.tensor(core_actions).long()
        actions[ai] = ai_actions.clone()
        actions = torch.cat((ids.reshape(-1,1), actions.reshape(-1,1)),1)
        return actions, ai_random_mask

    def weekly_ai_debug(self, mask, actions, ai_random, loss, ids, clusters, states):
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
        debug["friendly"] = torch.cat((
            ids.reshape(-1,1),
            scores.reshape(-1,1), 
            actions.reshape(-1,1),
            ai_random.reshape(-1,1),
            clusters.reshape(-1,1)),1)
        def entropy(x):
            x = torch.cat((x, torch.tensor([1596])))
            b = x.bincount().float()
            b[-1] -= 1
            b = torch.nn.functional.softmax(b, dim=0)
            b = torch.distributions.Categorical(b).entropy()
            return b
        debug["metrics"] = {
                "average_score": scores.float().mean(),
                "n_unique": actions.unique().size(0),
                "action_representation": actions.unique().size(0) / MessagesH.N,
                "action_entropy": entropy(actions)
                }
        return debug

    def generate_msg_qsn(self, current_actions, timeline, states):
        # generate weekly message/question file for users
        # message is based on current_actions
        # question is based on prev_actions
        previous_actions = self.stor["actions"].load_indexed_data(self.run_index-1)
        msg_qsn_mask = timeline[:,1] > 1
        if previous_actions is None or msg_qsn_mask.sum().item() == 0:
            return None
        current_actions = current_actions[msg_qsn_mask]
        file = [["ID", "M1_ID", "Q1_ID", "M2_ID", "Q2_ID"]]
        simul_resp = [["ID", "Q1_ID", "Q1_RESP", "Q2_ID", "Q2_RESP"]]
        for i in range(current_actions.size(0)):
            c, p = current_actions[i][1], previous_actions[i][1]
            row_id = current_actions[i][0].long().item()
            (m1, m2) = MessagesH.mid_lookup(c)
            (q1, q2) = QuestionsH.random_questions(MessagesH.sid_lookup(p))
            file.append([row_id, m1, q1, m2, q2])
            if self.simulate_responses:
                (s1, s2) = MessagesH.sid_lookup(p)
                r1 = states[i][s1-1]
                r2 = states[i][s2-1]
                if r1 % 1 == 0 and r1 != 3:
                    r1 += 0.001
                if r2 % 1 == 0 and r2 != 3:
                    r2 += 0.001
                r1 = torch.ceil(r1)
                r2 = torch.ceil(r2)
                r1 = torch.clip(r1, 0, 3).long().item()
                r2 = torch.clip(r2, 0, 3).long().item()
                # r1 = torch.randint(2, 4, (1,))[0].item()
                # r2 = torch.randint(2, 4, (1,))[0].item()
                simul_resp.append([row_id, q1, r1, q2, r2])
        if self.simulate_responses and not self.dry_run:
            self.stor["responses"].save_data(simul_resp, self.run_index) 
        return file

    def collect_responses(self):
        # collect and handle the responses for this week
        # if new file exists, load and align to ids vector
        prev_actions = self.stor["actions"].load_indexed_data(self.run_index-2)
        prev_clusters = self.stor["clusters"].load_indexed_data(self.run_index-2)
        resp = self.stor["responses"].load_indexed_data(self.run_index-1)
        if prev_actions is not None and resp is not None:
            if len(resp) == 0:
                return None, None
            resp = torch.tensor(resp).long()
            if self.simulate_responses:
                resp = resp[torch.randperm(resp.size(0))]
            algn = torch.zeros(resp.size(0)).long()
            for i in range(algn.size(0)):
                algn[i] = (resp[:,0] == prev_actions[i,0]).nonzero()[0]
            resp = resp[algn]
        return prev_actions, prev_clusters, resp
    
    def update_states(self, actions, clusters, responses, states, ids):
        # calculate this weeks state updates
        next_states = states.clone()
        transitions = []
        if actions is None or responses is None:
            return next_states, transitions
        update_idxs = torch.zeros(actions.size(0)).long()
        for i in range(actions.size(0)):
            update_idxs[i] = (ids == actions[i,0]).nonzero()[0]
        rewards = torch.zeros(actions.size(0)).long()
        for i, idx in enumerate(update_idxs):
            state = states[idx]
            action = actions[i,1]
            reward = 0
            next_state = next_states[idx].clone()
            resp = responses[i, [2,4]]
            sid = torch.tensor(MessagesH.sid_lookup(action)).long()
            for j in range(resp.size(0)):
                reward += torch.clip(resp[j] - state[sid[j]-1], 0)
                upd = torch.clip(resp[j] - state[sid[j]-1], 0)
                next_state[sid[j]-1] += upd / 2
            next_states[idx] = next_state
            rewards[i] = reward
            if resp.sum() == 0:
                continue
            cl = clusters[i]
            tr = [cl, state.clone(), action, reward, next_state.clone()]
            transitions.append(tr)
        return next_states, transitions

if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="yaml config file", required=True)
    parser.add_argument("-n", help="number of weeks to run", type=int, default=1)
    args = parser.parse_args()

    md = MDiabetes(args.f)
    for i in range(args.n):
        md.main()
        time.sleep(1)

