# https://github.com/ast0414/pointer-networks-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProcessorNetwork(nn.Module):
	def __init__(self, embedding_dim):
		super(ProcessorNetwork, self).__init__()
		self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

	def forward(self, embeddings, adjacency_matrix, cost_matrix):
		# Use adjacency matrix to filter neighbors and sum them
		degree = adjacency_matrix.sum(dim=-1).unsqueeze(2)	# (batch_size, nodes, 1)
		neighbor_embeddings = embeddings.unsqueeze(2) * cost_matrix.unsqueeze(-1)
		sum_neighbors = neighbor_embeddings.sum(dim=1) / degree
		# Concatenate original embeddings with their neighborhood sums
		concatenated = torch.cat([embeddings, sum_neighbors], dim=-1)
		return F.relu(self.fc(concatenated))


	# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
	"""
	``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
	masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
	``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
	``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
	broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
	unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
	do it yourself before passing the mask into this function.
	In the case that the input vector is completely masked, the return value of this function is
	arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
	of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
	that we deal with this case relies on having single-precision floats; mixing half-precision
	floats with fully-masked vectors will likely give you ``nans``.
	If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
	lower), the way we handle masking here could mess you up.  But if you've got logit values that
	extreme, you've got bigger problems than this.
	"""
	if mask is not None:
		mask = mask.float()
		while mask.dim() < vector.dim():
			mask = mask.unsqueeze(1)
		# vector + mask.log() is an easy way to zero out masked elements in logspace, but it
		# results in nans when the whole vector is masked.  We need a very small value instead of a
		# zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
		# just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
		# becomes 0 - this is just the smallest value we can actually use.
		vector = vector + (mask + 1e-45).log()
	return torch.nn.functional.log_softmax(vector, dim=dim)


	# Adopted from allennlp (https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py)
def masked_max(vector: torch.Tensor,
				mask: torch.Tensor,
				dim: int,
				keepdim: bool = False,
				min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
	"""
	To calculate max along certain dimensions on masked values
	Parameters
	----------
	vector : ``torch.Tensor``
		The vector to calculate max, assume unmasked parts are already zeros
	mask : ``torch.Tensor``
		The mask of the vector. It must be broadcastable with vector.
	dim : ``int``
		The dimension to calculate max
	keepdim : ``bool``
		Whether to keep dimension
	min_val : ``float``
		The minimal value for paddings
	Returns
	-------
	A ``torch.Tensor`` of including the maximum values.
	"""
	# one_minus_mask = (1.0 - mask).byte()
	one_minus_mask = (1.0 - mask).type(torch.bool)
	replaced_vector = vector.masked_fill(one_minus_mask, min_val)
	max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
	return max_value, max_index

def masked_sampling(vector: torch.Tensor,
				mask: torch.Tensor,
				min_val: float = -1e7) -> (torch.Tensor, torch.Tensor):
	"""
	To calculate max along certain dimensions on masked values
	Parameters
	----------
	vector : ``torch.Tensor``
		The vector to calculate max, assume unmasked parts are already zeros
	mask : ``torch.Tensor``
		The mask of the vector. It must be broadcastable with vector.
	min_val : ``float``
		The minimal value for paddings
	Returns
	-------
	A ``torch.Tensor`` of including the maximum values.
	"""
	one_minus_mask = (1.0 - mask).byte()
	replaced_vector = vector.masked_fill(one_minus_mask, min_val)
	log_p = torch.log_softmax(replaced_vector, dim=1)
	probs = log_p.exp()
	idxs = probs.multinomial(1)
	
	while one_minus_mask.gather(1, idxs).data.any():
		print(' [!] resampling due to race condition')
		idxs = probs.multinomial(1)
	return -1, idxs

def init_weights(models, weight_init_range=0.08):
	for model in models:
		for name, param in model.named_parameters():
			if 'bias' in name:
				nn.init.zeros_(param)
			else:
				nn.init.uniform_(param, -weight_init_range, weight_init_range)


class Encoder(nn.Module):
	def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True):
		super(Encoder, self).__init__()

		self.batch_first = batch_first
		self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
							batch_first=batch_first, bidirectional=False)
		init_weights([self.rnn])

	def forward(self, embedded_inputs):
		outputs, hidden = self.rnn(embedded_inputs)
		return outputs, hidden


class Attention(nn.Module):
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
		self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
		self.vt = nn.Linear(hidden_size, 1, bias=False)
		init_weights([self.W1, self.W2, self.vt])

	def forward(self, decoder_state, encoder_outputs, mask):
		# (batch_size, max_seq_len, hidden_size)
		encoder_transform = self.W1(encoder_outputs)

		# (batch_size, 1 (unsqueezed), hidden_size)
		decoder_transform = self.W2(decoder_state).unsqueeze(1)

		# 1st line of Eq.(3) in the paper
		# (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
		u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

		# softmax with only valid inputs, excluding zero padded parts
		# log-softmax for a better numerical stability
		log_score = masked_log_softmax(u_i, mask, dim=-1)

		return log_score


class PointerNet_TF(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_size, device='cpu', is_train=True):
		super(PointerNet_TF, self).__init__()

		# Embedding
		self.in_feature = input_dim
		self.embedding_dim = embedding_dim
		# Decoder
		self.hidden_size = hidden_size
		self.device = device

		self.num_layers = 1
		# training options
		self.is_train=is_train

		# We use an embedding layer for more complicate application usages later, e.g., word sequences.
		self.embedding = nn.Linear(in_features=self.in_feature, out_features=embedding_dim, bias=False)
		self.edge_embedding = ProcessorNetwork(embedding_dim)
		self.encoder = Encoder(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=self.num_layers,
								batch_first=True)
		self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
		self.attn = Attention(hidden_size=hidden_size)

		init_weights([self.embedding, self.decoding_rnn])

		for m in self.modules():
			if isinstance(m, nn.Linear):
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)

	def forward(self, x, init_x, target_lengths, max_seq_len, max_node_size, As, Cs, y=None, greedy=True):
		As = As.clone()
		batch_size = x.size(0)
		
		# Embedding
		embedded = self.embedding(x)    # (B, N, H)
		Cs = (1-Cs) * As	# Influence from distant nodes is minimal
		embedded = self.edge_embedding(embedded, As, Cs)
		encoder_outputs, encoder_hidden = self.encoder(embedded)

		encoder_h_n, encoder_c_n = encoder_hidden
		encoder_h_n = encoder_h_n.view(self.num_layers, 1, batch_size, self.hidden_size)
		encoder_c_n = encoder_c_n.view(self.num_layers, 1, batch_size, self.hidden_size)

		# init node extraction
		indices = init_x.view(-1, 1, 1).expand(-1, 1, encoder_outputs.size(-1))  # [500, 1, 1] -> [500, 1, 2]
		decoder_input = encoder_outputs.gather(1, indices).squeeze(1)	# (batch_size, self.hidden_size)
		decoder_hidden = (encoder_h_n[-1, 0, :, :], encoder_c_n[-1, 0, :, :])

		mask_tensor = self.cal_mask(batch_size, max_node_size, max_seq_len, target_lengths)

		pointer_log_scores = []
		pointer_argmaxs = []
		n_nodes= As.size(1)
		partial_sol = torch.zeros(batch_size, n_nodes).to(self.device)	# batch_size, node
		partial_sol.scatter_(1, init_x.view(-1,1), torch.ones(batch_size,1).to(self.device))
		sub_mask = torch.zeros(batch_size, n_nodes).to(self.device)	# B, N
		index_tensor = init_x.view(-1,1,1)	# B, 1, 1

		if self.is_train:
			# teacher forcing. x_t = y_(t-1)
			index_tensor = torch.cat([index_tensor, y.clone().view(batch_size,y.size(1),1)],dim=1)[:,:-1]
			index_tensor[index_tensor==-1]=0	# change pad to zero
			index_tensor = index_tensor.expand(batch_size, -1, self.hidden_size)
			decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor)
			for i in range(max_seq_len):
				sub_mask = mask_tensor[:, i, :].float()

				# h, c: (batch_size, hidden_size)
				h_i, c_i = self.decoding_rnn(decoder_input[:,i,:], decoder_hidden)
				decoder_hidden = (h_i, c_i)

				# Get a pointer distribution over the encoder outputs using attention
				# (batch_size, max_seq_len)
				log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
				pointer_log_scores.append(log_pointer_score)

				# Get the indices of maximum pointer
				_, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)

				pointer_argmaxs.append(masked_argmax)

		else:
			for i in range(max_seq_len):
				# use_edge_mask:
				add_mask = self.cal_seq_mask(As, index_tensor[:,:,0])
				sub_mask = ((sub_mask + add_mask) > 0).float()	# clipping (0 or 1)
				sub_mask[partial_sol==1] = 0		# rm partial solution 
				sub_mask[mask_tensor[:,i,:]==0] = 0	# rm longer than label length

				# h, c: (batch_size, hidden_size)
				h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

				# next hidden
				decoder_hidden = (h_i, c_i)

				# Get a pointer distribution over the encoder outputs using attention
				# (batch_size, max_seq_len)
				log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
				pointer_log_scores.append(log_pointer_score)

				if greedy:
					# Get the indices of maximum pointer
					_, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)

					pointer_argmaxs.append(masked_argmax)
				else:
					_, masked_argmax = masked_sampling(log_pointer_score, sub_mask)
					pointer_argmaxs.append(masked_argmax)

				# using prediction
				index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

				# (batch_size, hidden_size)
				decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)
				partial_sol.scatter_(1, index_tensor[:,:,0], torch.ones(batch_size,1).to(self.device))	# selected nodes


		pointer_log_scores = torch.stack(pointer_log_scores, 1)	# (B, L, N+1)
		pointer_argmaxs = torch.cat(pointer_argmaxs, 1)			# (B, L)

		return pointer_log_scores, pointer_argmaxs, mask_tensor

	def cal_mask(self, batch_size, max_node_size, max_seq_len, target_lengths):
		"""
		(B, L, N) rm padded nodes for loss
		"""
		range_tensor = torch.arange(max_seq_len, dtype=target_lengths.dtype).expand(batch_size, max_node_size, max_seq_len).transpose(2,1).to(self.device)
		each_len_tensor = target_lengths.view(-1, 1, 1).expand(batch_size, max_seq_len, max_node_size).to(self.device)
		mask_tensor = (range_tensor < each_len_tensor)
		return mask_tensor

	def cal_seq_mask(self, As, pointer_argmax):
		"""generate mask for not connected nodes

		Args:
			As (torch.Tensor): (B, N+1)-sized adjacency matrix
			pointer_argmax (torch.Tensor): (B, 1)-sized selected nodes
		"""
		batch_size, num_nodes = As.size(0), As.size(1)	# num_nodes = size
		# -> (B, 1, 1)
		pointer_argmax = pointer_argmax.view(batch_size, 1, 1)

		# (B, 1, N)
		sub_mask = As.gather(1, pointer_argmax.repeat(1, 1, num_nodes)).squeeze(1).to(self.device)
		return sub_mask

