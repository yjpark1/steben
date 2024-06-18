import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEmbedding(nn.Module):
    def __init__(self, p, k):
        super(GraphEmbedding, self).__init__()
        self.theta1 = nn.Parameter(torch.randn(p, 2))
        self.theta2 = nn.Parameter(torch.randn(p, k))

    def forward(self, s, t, x):
        sv_tv = torch.cat([s.unsqueeze(2), t.unsqueeze(2)], dim=2)
        mu_v = F.relu(torch.matmul(sv_tv, self.theta1.T) + torch.matmul(x, self.theta2.T))
        return mu_v


class ProcessorNetwork(nn.Module):
    def __init__(self, p):
        super(ProcessorNetwork, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(2 * p, 2 * p),
                nn.ReLU(),
                nn.Linear(2 * p, p)
            )

    def forward(self, mu_v, adjacency_matrix):
        # Use adjacency matrix to aggregate neighbor information
        aggregated_neighbors = torch.matmul(adjacency_matrix, mu_v)
        aggregated_neighbors -= adjacency_matrix.sum(dim=-1, keepdim=True) * mu_v
        
        # Apply linear transformation and ReLU activation
        mu_v_prime = F.relu(self.fc(torch.cat([mu_v, aggregated_neighbors], dim=-1)))
        return mu_v_prime

    
class DecoderNetwork(nn.Module):
    def __init__(self, p):
        super(DecoderNetwork, self).__init__()
        self.theta3 = nn.Parameter(torch.randn(2 * p))
        self.theta4 = nn.Parameter(torch.randn(p, p))
        self.theta5 = nn.Parameter(torch.randn(p, p))

    def forward(self, mu_v_prime):
        # Compute global state sum
        global_state = torch.mean(mu_v_prime, dim=1, keepdim=True)

        # Compute Q-value using theta4 and theta5
        global_feature = torch.matmul(global_state, self.theta4)
        local_feature = torch.matmul(mu_v_prime, self.theta5)

        # Concatenate global and local features
        combined_feature = torch.cat([global_feature.expand(mu_v_prime.shape), local_feature], dim=-1)
        
        # Compute Q-value
        q_value = torch.matmul(F.relu(combined_feature), self.theta3)        
        return q_value


class Vulcan(nn.Module):
    def __init__(self, p=64, k=2):
        super(Vulcan, self).__init__()
        self.encoder = GraphEmbedding(p, k)
        self.processor = ProcessorNetwork(p)
        self.decoder = DecoderNetwork(p)

    def forward(self, s, t, x, adjacency_mat):
        mu_v = self.encoder(s, t, x)
        mu_v_prime = self.processor(mu_v, adjacency_mat)
        q_value = self.decoder(mu_v_prime)
        return q_value
