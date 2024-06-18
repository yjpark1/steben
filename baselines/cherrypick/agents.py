import random
from collections import deque

import networkx as nx
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, network, args):
        self.network = network().to(args.device)
        self.target_network = network().to(args.device)
        self.replaybuffer = ReplayBuffer(args.buffer_size, args.batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.learning_rate)
        self.gamma = args.gamma
        self.target_update_freq = args.target_network_frequency
        self.batch_size = args.batch_size
        self.epsilon = args.start_e
        self.epsilon_min = args.end_e
        self.learning_starts = args.learning_starts
        self.args = args
        self.steps = 0
        ray.init(ignore_reinit_error=True)
        self.stateprocessor = ProcessStates(self.device)
        self.temperature = 0.1
    
    def get_masked_qvalues(self, state):
        s, t, x, adj = self.stateprocessor.process_single_state(state)
        qvalues = self.network(s, t, x, adj).detach()
        mask = self.stateprocessor._create_binary_vector(state.available_vertices, state.graph.number_of_nodes()) 
        mask = torch.from_numpy(mask).to(self.device)
        qvalues = qvalues + (mask - 1) * 1e9
        return qvalues
    
    def get_action(self, state, sampling=False):
        if type(state) is list:
            return self._get_batch_action_by_sampling(state)
        else:
            return self._get_action(state, sampling)
    
    @torch.no_grad()
    def _get_action(self, state, sampling=False):
        if random.random() < self.epsilon:
            selected_node = random.choice(state.available_vertices)
        else:
            qvalues = self.get_masked_qvalues(state)
            if sampling:
                probabilities = F.softmax(qvalues / self.temperature, dim=-1)
                selected_node = torch.multinomial(probabilities, 1).item()
            else:
                qval, selected_node = qvalues.max(1)
                selected_node = selected_node.cpu().item()
                # selected_node = min(state.available_vertices, key=lambda i: qvalues[0, i].item())
        return selected_node
    
    @torch.no_grad()
    def _get_batch_action_by_sampling(self, states):
        # only used for sampling inference
        states_ = self.process_batch(states)
        availables = self._get_mask_availables(states)
        
        # Compute Q targets for current states 
        qvalues = self.network(*states_).detach()
        qvalues += (availables - 1) * 1e9        
        probabilities = F.softmax(qvalues / self.temperature, dim=-1)
        selected_node = torch.multinomial(probabilities, 1).squeeze().cpu().numpy()
        return selected_node
        
    def _get_mask_availables(self, next_states_):
        availables = []
        for i, s in enumerate(next_states_):
            if i == 0: 
                num_vertices = s.graph.number_of_nodes()
            m = self.stateprocessor._create_binary_vector(s.available_vertices, num_vertices) 
            availables.append(m)
        availables = np.array(availables)
        availables = torch.from_numpy(availables).to(self.device)
        return availables
    
    def learn(self, episode):
        if len(self.replaybuffer) < self.batch_size:
            return None, None
        
        loss, q_expected = None, None
        if self.steps > self.learning_starts:
            if self.steps % self.args.train_frequency == 0:
                states_, actions, rewards, next_states_, dones = self.replaybuffer.sample()
                
                states = self.process_batch(states_)
                next_states = self.process_batch(next_states_)
                rewards = torch.from_numpy(rewards).type(torch.float32).to(self.device)
                dones = torch.from_numpy(dones).type(torch.float32).to(self.device)
                availables = self._get_mask_availables(next_states_)
                
                # Compute Q targets for current states 
                with torch.no_grad():
                    predicted_target_values = self.target_network(*next_states).detach()
                    # predicted_target_values *= availables
                    predicted_target_values += (availables - 1) * 1e9
                    target_max, _ = predicted_target_values.max(dim=1)
                    q_targets = rewards + self.gamma * target_max * (1 - dones)
                
                # Get expected Q values from local model
                q_expected = self.network(*states).gather(1, torch.from_numpy(actions).unsqueeze(1).to(self.device)).squeeze()
                
                # Compute loss
                loss = nn.MSELoss()(q_expected, q_targets)
                
                # Minimize the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
            # Update target network
            if self.steps % self.target_update_freq == 0:
                for target_network_param, q_network_param in zip(self.target_network.parameters(), self.network.parameters()):
                    target_network_param.data.copy_(
                        self.args.tau * q_network_param.data + (1.0 - self.args.tau) * target_network_param.data
                    )
        # Decay epsilon
        self.epsilon = linear_schedule(self.args.start_e, self.args.end_e, self.args.exploration_fraction * self.args.num_max_episodes, episode)
        self.steps += 1
        
        return loss, q_expected

    def process_batch(self, batch):
        S = []
        T = []
        X = []
        Adj = []
        
        for s, t, x, adj in map(self.stateprocessor.process_single_state, batch):
            S.append(s)
            T.append(t)
            X.append(x)
            Adj.append(adj)
        
        s = torch.cat(S).to(self.device)
        t = torch.cat(T).to(self.device)
        x = torch.cat(X).to(self.device)
        adj = torch.cat(Adj).to(self.device)
        return s, t, x, adj

    def _process_batch(self, batch):
        S = []
        T = []
        X = []
        Adj = []
        
        # Start tasks in parallel
        actor = ProcessStatesActor.remote()
        results = [actor.process_single_state.remote(state) for state in batch]
        
        # Collect the results
        for s, t, x, adj in ray.get(results):
            S.append(s)
            T.append(t)
            X.append(x)
            Adj.append(adj)
        
        s = torch.cat(S).to(self.device)
        t = torch.cat(T).to(self.device)
        x = torch.cat(X).to(self.device)
        adj = torch.cat(Adj).to(self.device)
        return s, t, x, adj
    


class ProcessStates:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
    
    def process_single_state(self, state):
        num_vertices = state.graph.number_of_nodes()
        sv = self._create_binary_vector(state.partial_solution, num_vertices)
        tv = self._create_binary_vector(state.graph.graph['Terminals']['terminals'], num_vertices)
        xv = np.array([v for _, v in sorted(state.distance.items(), key=lambda x: x[0])])
        adj = nx.adjacency_matrix(state.graph).todense()
        
        sv = np.array([sv], dtype=np.float32)
        tv = np.array([tv], dtype=np.float32)
        xv = np.array([xv], dtype=np.float32)
        adj = np.array([adj], dtype=np.float32)
        
        features = (
            torch.from_numpy(sv).to(self.device), 
            torch.from_numpy(tv).to(self.device),
            torch.from_numpy(xv).to(self.device),
            torch.from_numpy(adj).to(self.device)
        )
        
        return features
    
    @staticmethod
    def _create_binary_vector(indices, size):
        binary_vector = np.zeros(size, dtype=np.float32)
        # Set the specified indices to 1
        binary_vector[indices] = 1
        return binary_vector
    
@ray.remote
class ProcessStatesActor(ProcessStates):
    def __init__(self, device=torch.device('cpu')):
        super().__init__(device)
    
    
    