import json
from pathlib import Path
import torch
from storage import make_storage_group, load_yaml
from storage import Storage, MQDryStor, DBDryStor
from agent import ClusteredAgent
from content import StatesHandler, MessageHandler, QuestionHandler
from logger import MainLogger
from debugai import debugai

StatesH = StatesHandler()
MessagesH = MessageHandler()
QuestionsH = QuestionHandler()

def confirm_prod_run(config, override):
    if override:
        return
    dry = config["dry_run"]
    exp = config["storage"]["experiment"]
    if not dry and exp.lower() == "prod":
        print("-" * 50)
        print("CAUTION: YOU ARE PERFORMING A PRODUCTION RUN")
        print("ONLY DO THIS IF YOU ABSOLUTELY ARE SUPPOSED TO")
        who  = input("- ARE YOU JACK OR THANH? [yes/no] ").lower().strip()
        when = input("- IS IT WEDNESDAY? [yes/no] ").lower().strip()
        print("-" * 50)
        if who != "yes" or when != "yes":
            raise ValueError("Could not confirm production run")

class MDiabetes:

    def __init__(self, config_path, prod_confirm=False):
        self.config_path = config_path
        self.config = load_yaml(self.config_path)
        confirm_prod_run(self.config, prod_confirm)
        self.dry_run = self.config["dry_run"]
        self.simulate_responses = self.config["simulate_responses"]
        self.simulate_participants = self.config["simulate_participants"]
        self.stor = make_storage_group(**self.config["storage"])
        self.agent = None 
        self.run_index = -5

    def main(self):
        # read the MainLogger calls in this method to understand the flow
        self.run_index = self.stor["states"].count_files() + 1
        # for testing we might set a week index
        if self.config.get('force_week', False):
            self.run_index = self.config['week_idx']
        MainLogger("Starting week #", self.run_index)
        MainLogger("Dry run:", self.dry_run)
        MainLogger("Simulated Responses:", self.simulate_responses)
        MainLogger("Simulated Participants:", self.simulate_participants)
        MainLogger("Loading DQN agent")
        # initialize our clustered DQN agent for message selection
        self.agent = ClusteredAgent(self.config["cluster"], self.config["dqn"])
        self.agent.load_disk_repr(self.stor["dqns"], self.run_index-1)
        cluster_debug = None
        if not self.simulate_participants:
            # check for new participants and add them to our set
            # we need to know how long theyve been in the study (timeline), their unique ID,
            #  their cluster ID, their current behavior state, and some debugging info 
            #  from the cluster initialization
            MainLogger("Gathering real participants")
            timeline, ids, clusters, states, cluster_debug = self.gather_participants()
        else:
            MainLogger("Gathering simul participants")
            timeline, ids, clusters, states = self.gather_simulated_participants()
        MainLogger("# Participants:", states.size(0))
        weekly_loss = torch.tensor([])
        cluster_t_counts = {}
        if self.run_index == 1:
            # in the first week we warmup the DQN agent on all participants
            # and record this warmup loss
            MainLogger("Warming up agent")
            weekly_loss = self.warmup_agent(clusters, states)
        # need to know which participants receive which type of message
        #  optin participants do not get anything
        #  core participants get a non-AI preselected core message
        #  ai participants receive an AI selected message
        optin, core, ai = self.weekly_masks(timeline)
        # read the comments for each of these functions
        MainLogger("Generating weekly actions")
        actions, ai_random = self.weekly_actions(optin, core, ai, timeline, ids, clusters, states)
        MainLogger("Building message/question file")
        msg_qsn = self.generate_msg_qsn(actions, timeline, states)
        MainLogger("Checking for responses")
        prev_actions, prev_clusters, responses = self.collect_responses()
        MainLogger("updating states and adding transitions")
        next_states, transitions = self.update_states(prev_actions, prev_clusters, \
                                                      responses, states, ids)
        if self.run_index > 1 and len(transitions) > 0:
            # after the warmup week #1 we need to use the transitions built from
            #  responses we collect this week
            weekly_loss, cluster_t_counts = self.agent.weekly_training_update(transitions, self.run_index)
        MainLogger("analyzing ai performance")
        debug = debugai(ai, actions, ai_random, weekly_loss, ids, \
                clusters, states, cluster_debug, cluster_t_counts)
        MainLogger("transitions/cluster: ", cluster_t_counts)
        if not self.dry_run:
            # save all of our experiment data for analytics and use next week
            MainLogger("Saving data")
            self.stor["states"].save_data(next_states, self.run_index)
            self.stor["ids"].save_data(ids, self.run_index)
            self.stor["clusters"].save_data(clusters, self.run_index)
            if "read_actions" not in self.config['storage']:
                # this is some testing stuff, we sometimes want to use an
                #  existing set of actions and therefor do not want to save actions
                MainLogger("Saving actions")
                self.stor["actions"].save_data(actions, self.run_index)
            self.stor["timelines"].save_data(timeline, self.run_index)
            self.agent.save_disk_repr(self.stor["dqns"], self.run_index)
            self.stor["debugs"].save_data(debug, self.run_index)
            self.stor["outfiles"].save_data(msg_qsn, self.run_index)
            self.stor["yaml"].save_data(self.config_path, self.run_index)
        if self.dry_run:
            # for dry runs, we still want to save the AI performance data
            Path("dry_run_check").mkdir(exist_ok=True)
            MQDryStor.save_data(msg_qsn, self.run_index)
            DBDryStor.save_data(debug, self.run_index)
        MainLogger("="*5, "SUCCESS", "="*5)

    def gather_participants(self):
        # Check for new participants and add them to our set
        timeline = self.stor["timelines"].load_indexed_data(self.run_index-1)
        ids = self.stor["ids"].load_indexed_data(self.run_index-1)
        clusters = self.stor["clusters"].load_indexed_data(self.run_index-1)
        states = self.stor["states"].load_indexed_data(self.run_index-1)
        new_batch = self.stor["batches"].load_indexed_data(self.run_index)
        cluster_debug = None
        def modify_whatsapp(x):
            # helper function to parse the whatsapp numbers
            x = str(x)
            x = x[len(x)-10:]
            return int(x)
        if new_batch is not None:
            # we have a new batch, need to add them to our set
            all_whatsapps, all_states = StatesH.compute_states()
            new_states = None
            for new_glific_id, new_whatsapp in new_batch:
                new_whatsapp = modify_whatsapp(new_whatsapp)
                # find the new persons whatsapp, skip if not found
                where = (new_whatsapp == all_whatsapps).nonzero()
                if where.size(0) == 0:
                    continue
                # find the new glific ID, the new state, and build a timeline (ID, count) entry
                new_glific_id = torch.tensor([new_glific_id]).long()
                new_state = all_states[where[0]]
                new_tl = torch.cat((new_glific_id, torch.tensor([0]))).reshape(1,-1)
                # update our statesi/timeline dataset to include the new people
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
                # if this is the first week we need to set the clusters as well
                states = new_states
                MainLogger("Initializing clusters")
                cluster_debug = self.init_agent_clusters(modify_whatsapp, all_whatsapps, all_states)
                clusters = self.agent.assign_clusters(states)
            else:
                # otherwise just add the new data to the old data
                states = torch.cat((states, new_states))
                new_clusters = self.agent.assign_clusters(new_states)
                clusters = torch.cat((clusters, new_clusters))
        timeline[:,1] += 1  # increase all participants timeline count by 1
        return timeline, ids, clusters, states, cluster_debug

    def init_agent_clusters(self, mwa, all_whatsapps, all_states):
        # Use all baseline states of participants to calculate the cluster centers
        new_batch = Storage.load_csv("arogya_content/all_ai_participants.csv")
        idxs = []
        for gid, wa in new_batch:
            # some participants do not have assigned glific IDs, so we ignore them
            wa = mwa(wa)
            where = (wa == all_whatsapps).nonzero()
            if where.size(0) == 0:
                continue
            idxs.append(where)
        idxs = torch.tensor(idxs).long()
        self.agent.init_clusters(all_states[idxs])

    def gather_simulated_participants(self):
        # testing
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
        # Run the DQN warmup algorithm using all baseline states
        targets = torch.zeros(states.size(0), MessagesH.N)
        mesage_sids = []
        for col in range(MessagesH.N):
            sids = MessagesH.sid_lookup(col)
            mesage_sids.append(sids)
        for row in range(states.size(0)):
            # for each action, the initial/warmup Q-value is the 
            #  difference between the max state value and the participants
            #  state value
            # in hindsight this does not account for future reward, so the 
            #  warmup Q-values are smaller than they should be
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
                # new participants
                optin_mask[idx] = True
            elif tl.long().item() in MessagesH.core_timeline_map:
                # core participants
                core_mask[idx] = True
            else:
                # else
                ai_mask[idx] = True
        return (optin_mask, core_mask, ai_mask)

    def weekly_actions(self, optin, core, ai, timeline, ids, clusters, states):
        # determine which actions to take this week
        # optin group gets random core action
        # core group needs scheduled action
        # ai group gets input to dqn
        if "read_actions" in self.config['storage']:
            # some tests we want to use pregenerated actions, so instead of
            #  generating them now we read from a file
            MainLogger("Reading actions")
            last = self.stor["debugs"].load_indexed_data(self.run_index)
            actions = last["friendly"][:,2]
            ai_random_mask = last["friendly"][:,3]
            return actions, ai_random_mask
        actions = torch.zeros(states.size(0)).long()
        # choose random core messages to be recorded (but not sent) to first-time participants
        optin_actions = MessagesH.random_core_actions(optin.sum().item())
        # choose preselected core messages for certain users
        core_actions = MessagesH.scheduled_core_actions(timeline[core])
        # generate dynamic AI messages for rest of participants
        ai_random_mask, ai_actions, ai_clusters = self.agent.choose_actions(clusters[ai], states[ai])
        actions[optin] = torch.tensor(optin_actions).long() 
        actions[core] = torch.tensor(core_actions).long()
        actions[ai] = ai_actions.clone()
        actions = torch.cat((ids.reshape(-1,1), actions.reshape(-1,1)),1)
        return actions, ai_random_mask

    def generate_msg_qsn(self, current_actions, timeline, states):
        # Generate a formatted M/Q file for all valid participants
        # load the previous actions which will prompt this weeks questions
        previous_actions = self.stor["actions"].load_indexed_data(self.run_index-1)
        # no ai questions are sent in week 1. The study is 32 weeks long
        msg_qsn_mask = (timeline[:,1] > 1) & (timeline[:,1] <= 32)
        if previous_actions is None or msg_qsn_mask.sum().item() == 0:
            return None
        # perform this only for valid participants
        current_actions = current_actions[msg_qsn_mask]
        file = [["PARTICIPANT_ID", "M1_ID", "Q1_ID", "M2_ID", "Q2_ID"]]
        simul_resp = [["PARTICIPANT_ID", "Q1_ID", "Q1_RESP", "Q2_ID", "Q2_RESP"]]
        # current actions has the columns (id, action_id)
        for i in range(current_actions.size(0)):
            # current action is the messages, previous actions used for questions
            c, p = current_actions[i][1], previous_actions[i][1]
            row_id = current_actions[i][0].long().item()
            (m1, m2) = MessagesH.mid_lookup(c)  # lookup the message IDs for this action ID
            # get two random questions for the state elements from the previous action
            (q1, q2) = QuestionsH.random_questions(MessagesH.sid_lookup(p)) 
            file.append([row_id, m1, q1, m2, q2])  # record in file to be uploaded
            if self.simulate_responses:
                # this is for testing, make some simulated responses
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
                simul_resp.append([row_id, q1, r1, q2, r2])
        if self.simulate_responses and not self.dry_run:
            self.stor["responses"].save_data(simul_resp, self.run_index) 
        return file

    def collect_responses(self):
        # load the previous actions which prompted these responses
        prev_actions = self.stor["actions"].load_indexed_data(self.run_index-2)
        # Load the clusters of participants who were sent M/Q
        prev_clusters = self.stor["clusters"].load_indexed_data(self.run_index-2)
        # Attempt to load a new batch of responses
        resp = self.stor["responses"].load_indexed_data(self.run_index-1)
        cresp = None
        if prev_actions is not None and resp is not None:
            # if we have actions and responses to these actions, we can 
            #  build an aligned dataset of the participant responses
            MainLogger("handling responses")
            cresp = torch.zeros(prev_actions.size(0), 5).long()
            if len(resp) == 0:
                return None, None
            resp = torch.tensor(resp).long()
            if self.simulate_responses:
                MainLogger("Shuffling simulated responses")
                resp = resp[torch.randperm(resp.size(0))]
            # loop over each previous action and align the 
            #  IDs of the action file and response file
            for i in range(prev_actions.size(0)):
                idx = (resp[:,0] == prev_actions[i,0]).nonzero()
                if idx.size(0) == 0:
                    cresp[i,0] = prev_actions[i,0]
                else:
                    cresp[i] = resp[idx[0]]
        return prev_actions, prev_clusters, cresp
    
    def update_states(self, actions, clusters, responses, states, ids):
        # Update participant states based on responses, and build transitions
        #  to be used in the DQN training
        next_states = states.clone()
        transitions = []
        if actions is None or responses is None:
            # nothing to do if we didnt send actions or receive responses
            # such as in week 1 or 2
            return next_states, transitions
        # the data files (messages, responses, current states) are not aligned
        #  by participant id. need to create some index array to align all of them
        update_idxs = torch.zeros(actions.size(0)).long()
        for i in range(actions.size(0)):
            update_idxs[i] = (ids == actions[i,0]).nonzero()[0]
        rewards = torch.zeros(actions.size(0)).long()
        updated = torch.zeros_like(ids).bool()
        # loop over all of the indexes of states we need to update
        for i, idx in enumerate(update_idxs):
            state = states[idx] # pull the participants state
            updated[idx] = True
            action = actions[i,1]  # find the action we sent them
            crewards = [None, None]
            reward = 0
            next_state = next_states[idx].clone() # copy their current state to update
            resp = responses[i, [2,4]]  # select their responses from the response set
            questions = responses[i,[1,3]]  # select the questions we asked
            sid = torch.tensor(MessagesH.sid_lookup(action)).long()  # find the element IDs
            # loop over their responses and calculate the reward and update
            # the reward is the difference between their state elem value and their response
            # if they respond, their state elem gets updated to the response
            for j in range(resp.size(0)):
                crewards[j] = torch.clip(resp[j] - state[sid[j]-1], 0, 3)
                if resp[j] != 0:
                    next_state[sid[j]-1] = resp[j]
            # update the state in our dataset
            next_states[idx] = next_state.clone()
            reward = sum(crewards)/len(crewards)
            rewards[i] = reward
            # do not collect a transition if they do not respond to any question
            if resp.sum() == 0:
                continue
            cl = clusters[i]
            # build a transition for this participant
            tr = [cl, state.clone(), action, reward, next_state.clone()]
            transitions.append(tr)
        return next_states, transitions

if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="yaml config file", required=True)
    parser.add_argument("-n", help="number of weeks to run", type=int, default=1)
    parser.add_argument("--prod", "-p", dest="prod", action="store_true")
    args = parser.parse_args()

    for i in range(args.n):
        md = MDiabetes(args.f, args.prod)
        md.main()
        time.sleep(1)
    print("mdiabetes done :)")


