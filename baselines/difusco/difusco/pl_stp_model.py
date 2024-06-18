import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from data.stp_graph_dataset import STPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.stp_utils import STPEvaluator, TwoApproximation, stp_merge_tours, Cherrypick
import time

class STPModel(COMetaModel):
    def __init__(self,
                 param_args=None):
        super(STPModel, self).__init__(param_args=param_args)

        self.train_dataset = STPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.training_split),
            sparse_factor=self.args.sparse_factor,
        )
        self.test_dataset = STPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse_factor=self.args.sparse_factor, test_cost=self.args.test_cost,
        )

        self.validation_dataset = STPGraphDataset(
            data_file=os.path.join(self.args.storage_path, self.args.validation_split),
            sparse_factor=self.args.sparse_factor, test_cost=self.args.test_cost,
        )
        if self.args.wandb_logger_name:
            self.logger_dir = self.args.logger_dir + self.args.wandb_logger_name + '/'
        
        if not os.path.exists(self.logger_dir):
            os.makedirs(self.logger_dir)

        self.gt_cost = []
        self.none_cost = []
    
    def forward(self, x, adj, t, edge_index, edge_cost):
        return self.model(x, t, adj, edge_index, edge_cost)

    def categorical_training_step(self, batch, batch_idx):
        _, points, adj_matrix, _, terminals, edge_cost, _ = batch
        # Sample from diffusion
        t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
        adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
        
        xt = self.diffusion.sample(adj_matrix_onehot, t)
        xt = xt * 2 -1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
        
        # Denoise
        x0_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            None,
            edge_cost,
        )
        
        # Compute loss
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, adj_matrix.long())
        self.log("train/loss", loss)
        
        return loss

    def gaussian_training_step(self, batch, batch_idx):
        if self.sparse:
        # TODO: Implement Gaussian diffusion with sparse graphs
            raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
        _, points, adj_matrix, _, terminals, edge_cost, _ = batch

        adj_matrix = adj_matrix * 2 - 1
        adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
        # Sample from diffusion
        t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
        xt, epsilon = self.diffusion.sample(adj_matrix, t)

        t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
        # Denoise
        epsilon_pred = self.forward(
            points.float().to(adj_matrix.device),
            xt.float().to(adj_matrix.device),
            t.float().to(adj_matrix.device),
            None,
            edge_cost,
        )
        epsilon_pred = epsilon_pred.squeeze(1)

        # Compute loss
        loss = F.mse_loss(epsilon_pred, epsilon.float())
        self.log("train/loss", loss)
        
        return loss

    def training_step(self, batch, batch_idx):
        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            return self.categorical_training_step(batch, batch_idx)

    def categorical_denoise_step(self, points, xt, t, device, edge_cost, edge_index=None, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                points.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
                edge_cost,
            )

            if not self.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
        
            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt

    def gaussian_denoise_step(self, points, xt, t, device, edge_cost, edge_index=None, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                points.float().to(device),
                xt.float().to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
                edge_cost,
            )
            pred = pred.squeeze(1)
            xt = self.gaussian_posterior(target_t, t, pred, xt)
            return xt

    def test_step(self, batch, batch_idx, split='test'):
        edge_index = None
        np_edge_index = None
        device = batch[-1].device
        real_batch_idx, points, adj_matrix, gt_tour, terminals, edge_cost, gt_cost = batch
        np_points = points.cpu().numpy()[0]
        np_gt_tour = gt_tour.cpu().numpy()[0]
        np_gt_tour = [(np_gt_tour[i], np_gt_tour[i+1]) for i in range(0, len(np_gt_tour), 2) if np_gt_tour[i] != -1 and np_gt_tour[i+1] != -1]

        masked_tours = []
        none_cost_tours = []
        only_cost_tours = []
        ns, merge_iterations = 0, 0

        if self.args.parallel_sampling > 1:
            if not self.sparse:
                points = points.repeat(self.args.parallel_sampling, 1, 1)
            else:
                points = points.repeat(self.args.parallel_sampling, 1)
                edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

        for _ in range(self.args.sequential_sampling):
            xt = torch.randn_like(adj_matrix.float())
        
            if self.args.parallel_sampling > 1:
                if not self.sparse:
                    xt = xt.repeat(self.args.parallel_sampling, 1, 1)
                else:
                    xt = xt.repeat(self.args.parallel_sampling, 1)
                xt = torch.randn_like(xt)
        
            if self.diffusion_type == 'gaussian':
                xt.requires_grad = True
            else:
                xt = (xt>0).long()
            
            if self.sparse:
                xt = xt.reshape(-1)

            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

            # Diffusion iterations
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)

                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(
                        points, xt, t1, device, edge_cost, edge_index, target_t=t2)
                else:
                    xt = self.categorical_denoise_step(
                        points, xt, t1, device, edge_cost, edge_index, target_t=t2)
                
            if self.diffusion_type == 'gaussian':
                adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().detach().numpy() + 1e-6

            if self.args.save_numpy_heatmap:
                self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

            # Get Tours
            solver = Cherrypick(adj_mat, terminals, edge_cost, alpha=0.9)
            solver.solve()
            # solver = TwoApproximation(adj_mat, terminals, edge_cost, alpha=0.9)
            # solver.solve()

            none_cost_tours.append(list(solver.solution_none_cost.edges()))

        # Get Cost 
        stp_solver = STPEvaluator()
        if split=='val':
            gt_cost = np.array(stp_solver.evaluate(np_gt_tour, edge_cost))

        total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
        all_none_cost = [stp_solver.evaluate(none_cost_tours[i], edge_cost) for i in range(total_sampling)]

        best_none_cost = np.min(np.array(all_none_cost))
        # none_cost = stp_solver.evaluate(list(solver.solution_none_cost.edges()), edge_cost)

        self.gt_cost.append(gt_cost.item())
        self.none_cost.append(best_none_cost)

        with open(f'{self.logger_dir}gt_cost_{device.index}.pkl', 'wb') as f:
            pickle.dump(self.gt_cost, f)
        with open(f'{self.logger_dir}none_cost_{device.index}.pkl', 'wb') as f:
            pickle.dump(self.none_cost, f)


        # logger 및 Evaluator 구현
        metrics = {
            f"{split}/gt_cost": 0,
            f"{split}/2opt_iterations": ns,
            f"{split}/merge_iterations": merge_iterations,
        }
        self.log(f"{split}/solved_cost", best_none_cost, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"output_cost", best_none_cost)
        return metrics

    def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
        if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
            raise NotImplementedError("Save numpy heatmap only support single sampling")
        exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
        heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
        rank_zero_info(f"Saving heatmap to {heatmap_path}")
        os.makedirs(heatmap_path, exist_ok=True)
        real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
        np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
        np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')