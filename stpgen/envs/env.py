import gymnasium
from gymnasium import spaces
import networkx as nx
import numpy as np

class STPEnvironment(gymnasium.Env):
    def __init__(self, graph, terminal_nodes, terminal_reward=-1, step_reward=-0.1):
        super(STPEnvironment, self).__init__()

        self.graph = graph
        self.terminal_nodes = terminal_nodes
        self.terminal_reward = terminal_reward
        self.step_reward = step_reward

        # Define action and observation space
        self.action_space = spaces.Discrete(len(graph.nodes))
        self.observation_space = spaces.MultiBinary(len(graph.nodes))

        # Initialize episode variables
        self.current_node = None
        self.visited_nodes = set()

    def reset(self):
        # Reset episode variables
        self.current_node = np.random.choice(self.terminal_nodes)
        self.visited_nodes = set()

        # Reset environment state (observation)
        state = self._get_state()

        return state

    def step(self, action):
        # Move to the next node
        self.current_node = action
        self.visited_nodes.add(action)

        # Update environment state (observation)
        state = self._get_state()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if the episode is done
        done = self._is_terminal()

        return state, reward, done, {}

    def _get_state(self):
        # Encode the state (observation) as a binary vector indicating visited nodes
        state = np.zeros(len(self.graph.nodes), dtype=np.int)
        state[list(self.visited_nodes)] = 1
        return state

    def _calculate_reward(self):
        # Custom reward function based on the number of visited nodes
        if self._is_terminal():
            return self.terminal_reward
        else:
            return self.step_reward

    def _is_terminal(self):
        # Check if all terminal nodes are visited
        return set(self.terminal_nodes).issubset(self.visited_nodes)

# Example usage:
# Create a graph using NetworkX with specified terminal nodes
num_nodes = 10
G = nx.complete_graph(num_nodes)
terminal_nodes = [0, 2, 4]

# Create an instance of the STP environment
env = STPEnvironment(G, terminal_nodes)

# Reset the environment
state = env.reset()
print("Initial state:", state)

# Perform a few steps in the environment
for _ in range(4):
    action = env.action_space.sample()  # Random action
    next_state, reward, done, _ = env.step(action)
    print("Action:", action, "Next state:", next_state, "Reward:", reward, "Done:", done)
