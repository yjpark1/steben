import math

import torch
from torch_scatter import segment_sum_csr, segment_max_csr

FLOAT_NEG_INF = -float('inf')
FLOAT_NEG_INF_THRESH = -1e16
FLOAT_EPS = 1e-6

SLICE_NO_FIRST = slice(1, None)
SLICE_NO_LAST = slice(None, -1)


class TspBaseSampler:
    def __init__(self, x, deg, edge_v, sample_size, optionsF):
        self.n_nodes = x.size(0)
        self.n_edges = edge_v.size(-1)
        self.sample_size = sample_size
        self.n_cands = self.n_nodes
        self.x = x
        self.deg = deg
        self.edge_v = edge_v
        self.optionsF = optionsF
        self.device = optionsF.device
        self.optionsI = torch.zeros(0, dtype=torch.int64, device=self.device)
        self.mask = torch.ones((sample_size, self.n_nodes), dtype=torch.bool, device=self.device)
        self.cands = torch.arange(self.n_nodes, dtype=torch.int64, device=self.device).repeat(sample_size, 1)
        self.s_grp = torch.arange(sample_size, dtype=torch.int64, device=self.device)
        self.e_ofs = torch.arange(self.n_nodes * sample_size, dtype=torch.int64, device=self.device)
        self.par_ptr = torch.cat((torch.zeros(1, dtype=torch.int64, device=self.device), deg[:-1]), 0).cumsum(0)

    def init(self):
        raise NotImplementedError

    def start(self, v0):
        self.mask.scatter_(-1, v0.unsqueeze(-1), 0)

    def sample_cands(self, i, u, v, s_idx):
        s_size = s_idx.size(0)
        if s_size > 0:
            n_next = self.n_nodes - i
            us = u.index_select(0, s_idx).unsqueeze(1).expand(s_size, n_next).flatten()
            cs = self.cands.index_select(0, s_idx)
            vs = cs.masked_select(self.mask.index_select(0, s_idx).gather(1, cs))
            dist = torch.pairwise_distance(self.x.index_select(0, us), self.x.index_select(0, vs)).reshape(s_size,
                                                                                                           n_next)
            vs_idx = dist.argmin(1, keepdim=True)
            v.index_put_((s_idx,), vs.reshape(s_size, n_next).gather(1, vs_idx).squeeze(1))

    def compute_scores(self, e_msk, e_idx):
        raise NotImplementedError

    def update(self, v_idx, s_idx, e_idx, e_msk, e_grp, e_ptr):
        pass

    def transit(self, i, u, v):
        self.mask.scatter_(-1, v.unsqueeze(-1), 0)
        if (self.n_nodes - i - 1) * 2 < self.n_cands and i < self.n_nodes - 1:
            self.n_cands = self.n_nodes - i - 1
            self.cands = self.cands.masked_select(self.mask.gather(-1, self.cands)).view(self.sample_size, self.n_cands)

    def finalize(self, u, v0):
        pass

    def result(self):
        return []

    def sample(self):
        v0 = self.init()  # 시작점
        u = v0  # 현재 노드 저장
        v = torch.empty_like(u)
        self.start(v0)  # 방문기록 (mask)에 값 넣어줌
        e_wid = torch.empty(self.sample_size + 1, dtype=torch.int64, device=self.device)
        e_wid[0] = 0
        for i in range(1, self.n_nodes):
            e_deg = self.deg.index_select(0, u)  # 현재노드의 degree
            e_grp = self.s_grp.repeat_interleave(e_deg)  # 엣지 표현 위해 각 노드의 degree만큼 확장
            e_wid[1:] = e_deg
            e_ptr = e_wid.cumsum(0)  # 각 노드의 엣지 시작 끝 index
            e_idx = (self.par_ptr.index_select(0, u) - e_ptr[:-1]).repeat_interleave(e_deg) + self.e_ofs[:e_grp.size(0)]
            # 전체 엣지 리스트에서 인덱스를 찾기 위해 수행.
            # e_idx에는 현재 노드에 해당하는 엣지의 인덱스를 저장 (edge list에서)

            e_msk = self.mask.index_select(0, e_grp).gather(1, self.edge_v.index_select(0, e_idx).unsqueeze(-1)).squeeze(-1)
            # self.edge_v에는 각 엣지의 target node가 들어있음.
            # self.edge_v.index_select(0, e_idx)를 하게 되면 엣지 인덱스에 해당하는 target node 를 가져옴
            # target node가 갈 수 있는지 마스킹

            scores = self.compute_scores(e_msk, e_idx)
            # scores 는
            v_scr, v_idx = segment_max_csr(scores, e_ptr)
            # e_ptr로 분할된 구역에서 column-wise로 max 값을 취해줌
            s_msk = (v_scr > FLOAT_NEG_INF_THRESH) & (v_idx != e_idx.size(0))
            s_idx = s_msk.nonzero()
            self.update(v_idx.masked_select(s_msk), s_idx, e_idx, e_msk, e_grp, e_ptr)
            v.index_put_((s_idx.squeeze(-1),), self.edge_v.index_select(0, e_idx.index_select(0, v_idx.index_select(0, s_idx.squeeze(-1)))))
            # v.index_put_((s_idx, ), self.edge_v.index_select(0, e_idx.index_select(0, v_idx[s_idx[:, 0], s_idx[:, 1]])).unsqueeze(-1))
            self.sample_cands(i, u, v, (~s_msk).nonzero().squeeze(-1))
            self.transit(i, u, v)
            u = v.clone()
        self.finalize(u, v0)
        return self.result()


class TspSampler(TspBaseSampler):
    def __init__(self, x, deg, edge_v, sample_size, optionsF):
        super().__init__(x, deg, edge_v, sample_size, optionsF)
        self.y = torch.zeros(sample_size, dtype=optionsF.dtype, device=self.device)
        self.tours = torch.empty((self.n_nodes, sample_size), dtype=torch.int64, device=self.device)

    def start(self, v0):
        super().start(v0)
        self.tours[0] = v0

    def transit(self, i, u, v):
        super().transit(i, u, v)
        self.y.add_(torch.pairwise_distance(self.x.index_select(0, u), self.x.index_select(0, v)))
        self.tours[i] = v

    def finalize(self, u, v0):
        super().finalize(u, v0)
        self.y.add_(torch.pairwise_distance(self.x.index_select(0, u), self.x.index_select(0, v0)))

    def result(self):
        result = super().result()
        result.append(self.y)
        result.append(self.tours.t())
        return result


class TspGreedySampler(TspSampler):
    def __init__(self, x, deg, edge_v, par, sample_size):
        super().__init__(x, deg, edge_v, sample_size, torch.zeros(0, dtype=par.dtype, device=par.device))
        self.par = (par - par.mean()).clone()
        self.neg_inf = torch.full((1,), FLOAT_NEG_INF, dtype=par.dtype, device=par.device)

    def init(self):
        assert self.n_nodes >= self.sample_size
        return torch.randperm(self.n_nodes, dtype=torch.int64, device=self.device)[:self.sample_size]

    def compute_scores(self, e_msk, e_idx):
        return torch.where(e_msk, self.par.index_select(0, e_idx), self.neg_inf)


class TspSoftmaxSampler(TspGreedySampler):
    def __init__(self, x, deg, edge_v, par, sample_size, y_bl):
        super().__init__(x, deg, edge_v, par, sample_size)
        self.y_bl = y_bl

    def init(self):
        return torch.randint(self.n_nodes, (self.sample_size,), dtype=torch.int64, device=self.device)

    def compute_scores(self, e_msk, e_idx):
        self.par_e = super().compute_scores(e_msk, e_idx)
        return self.par_e - torch.empty_like(self.par_e).exponential_().log()


class TspSoftmaxGradSampler(TspSoftmaxSampler):
    def __init__(self, x, deg, edge_v, par, sample_size, y_bl):
        super().__init__(x, deg, edge_v, par, sample_size, y_bl)
        self.gi_i_idx = []
        self.gi_s_idx = []
        self.gi_p_idx = []
        self.gi_p_grp = []
        self.gi_probs = []

    def update(self, v_idx_, s_idx_, e_idx_, e_msk_, e_grp_, e_ptr_):
        i_idx_ = e_idx_.index_select(0, v_idx_)
        e_msk_idx_ = e_msk_.nonzero(as_tuple=True)[0]
        p_idx_ = e_idx_.index_select(0, e_msk_idx_)
        p_grp_ = e_grp_.index_select(0, e_msk_idx_)
        logits = self.par_e - segment_max_csr(self.par_e, e_ptr_)[0].index_select(0, e_grp_)
        par_exp_ = logits.exp()
        p_denom_ = segment_sum_csr(par_exp_, e_ptr_)
        probs_ = par_exp_.index_select(0, e_msk_idx_) / p_denom_.index_select(0, p_grp_)
        self.gi_i_idx.append(i_idx_)
        self.gi_s_idx.append(s_idx_)
        self.gi_p_idx.append(p_idx_)
        self.gi_p_grp.append(p_grp_)
        self.gi_probs.append(probs_)

    def result(self):
        result = super().result()
        result.pop()
        grad = torch.zeros_like(self.par, dtype=self.optionsF.dtype, device=self.device)
        coefs = self.y
        isnan = False
        if isinstance(self.y_bl, float):
            if math.isnan(self.y_bl):
                isnan = True
        elif torch.isnan(self.y_bl):
            isnan = True
        if isnan:
            coefs -= self.y.mean()
        else:
            coefs -= self.y_bl
        i_idx = torch.cat(self.gi_i_idx)
        s_idx = torch.cat(self.gi_s_idx).squeeze(-1)
        p_idx = torch.cat(self.gi_p_idx).squeeze(-1)
        p_grp = torch.cat(self.gi_p_grp).squeeze(-1)
        probs = torch.cat(self.gi_probs).squeeze(-1)
        grad.scatter_add_(0, i_idx, coefs.index_select(0, s_idx))
        grad.scatter_add_(0, p_idx, coefs.index_select(0, p_grp) * -probs)
        grad /= self.sample_size
        result.append(grad)
        return result


def tsp_greedy_py(x, deg, edge_v, par, sample_size):
    return TspGreedySampler(x, deg, edge_v, par, sample_size).sample()


def tsp_softmax_py(x, deg, edge_v, par, sample_size, y_bl):
    return TspSoftmaxSampler(x, deg, edge_v, par, sample_size, y_bl).sample()


def tsp_softmax_grad_py(x, deg, edge_v, par, sample_size, y_bl):
    return TspSoftmaxGradSampler(x, deg, edge_v, par, sample_size, y_bl).sample()
