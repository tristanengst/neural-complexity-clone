import torch
from torch.utils.data import Sampler
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm_notebook as tqdm

class IntelligentSampler(Sampler):

	def __init__(self, dataset):
		""" A Sampler with infrastructure for finding out exactly what was
		sampled. This will not work as a sampler but it's nice to factor code
		into it.

		Args:
		dataset	-- the dataset subclasses will sample from
		"""
		print("Setting up the sampler, this could take a minute...")
		self.dataset = dataset
		self.num_classes, cts, stc = 0, {}, []
		for i,(_,c) in self.dataset:
			stc.append(c)
			if c not in cts:
				cts[c], self.num_classes = [i], self.num_classes + 1
			else:
				cts[c].append(i)

		self.class_to_subset = {c : torch.tensor(cts[c]) for c in cts}
		self.subset_to_class = torch.tensor(stc, requires_grad=False)
		self.subset_to_times = torch.ones(len(dataset), requires_grad=False)

	def get_example_counts(self):
		""" Returns a list [result] in which result[i] is the number of times
		the [ith] class has been sampled.
		"""
		result = []
		for c in self.class_to_subset:
			num_subset_times = 0
			for subset_index in self.class_to_subset[c]:
				num_subset_times += self.subset_to_times[subset_index].item()
			result.append(num_subset_times)
		return result

	def __iter__(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

class BanditSampler(IntelligentSampler):

	def __init__(self, dataset, num_batches, beta=1.414, decay=.6):
		"""
		Args:
		dataset		-- the dataset to sample from
		num_batches	-- the number of batches to return per call to __iter__()
		beta		-- the constant by which to multiply the exploration term
		decay		-- the rate at which to decay old losses
		"""
		IntelligentSampler.__init__(self, dataset)
		self.num_batches 	= num_batches
		self.subset_to_loss = torch.zeros(len(dataset), requires_grad=False)
		self.ucb_values  	= torch.ones(len(dataset), requires_grad=False)
		self.beta 		 	= beta
		self.decay 		 	= decay
		self.total_times 	= 1.

	def update_sampler(self, outputs, labels, indices) -> None:
		"""Updates the sampler's data.

		Args:
		outputs	-- a tensor of the model's outputs
		labels	-- a tensor of labels for [outputs]
		indices	-- a tensor of indices for [outputs]
		"""
		def compute_update(o, i):
			""" [o] is a model's output on the [ith] example in [labels]"""
			return ((sum(o) - o[labels[i]]) / (self.num_classes - 1)) + 1 - o[labels[i]]

		outputs = torch.sigmoid(outputs)
		bandit_losses = torch.tensor([compute_update(o, i) for i, o in enumerate(outputs)])

		self.total_times += len(indices)
		for i, loss in enumerate(bandit_losses):
			index = indices[i]
			self.subset_to_loss[index] = loss + self.subset_to_loss[index] * self.decay
			self.subset_to_times[index] += 1

		explore = torch.sqrt(torch.log(self.subset_to_times) / (self.total_times + 1))
		reward = self.subset_to_loss / self.subset_to_times
		self.ucb_values = reward + self.beta * explore

	def __iter__(self):
		"""Returns an iterator over the data's indices"""
		indices = torch.multinomial(self.ucb_values, self.num_batches, replacement=True)
		return iter(indices)

	def __len__(self):
		"Returns the number of training examples returned by __iter__()"
		return self.num_batches

	def get_class_data(self):
		"""Returns a list with one sublist for each class, with a structure
		[[average reward, average explore]]"""
		result = []
		explore_term = torch.sqrt(torch.log(self.subset_to_times) / self.total_times)
		for c in self.class_to_subset:
			indices = self.class_to_subset[c]
			c_loss = torch.tensor([self.subset_to_loss[i] for i in indices])
			c_times = torch.tensor([self.subset_to_times[i] for i in indices])
			rewards = sum(c_loss / c_times) / len(indices)
			explore = self.beta * sum(torch.sqrt(torch.log(c_times) / self.total_times)) / len(indices)
			result.append([round(rewards.item(), 2), round(explore.item(), 2)])
		return result

class MarginBanditSampler(BanditSampler):
	""" Samples proportionally to a spicy UCB based on decision boundaries.

	In our work, examples are sampled according to probabilities derived from

			reward + beta * needed exploration

	normalized across all samples, where needed exploration is computed from the
	sampler's past samples, and reward is computed from the outputs of
	the model. This allows examples that are furthest towards the wrong side of
	the most decision boundaries between their class and others more often, and
	those that are at least gamma on the right side of the decision boundaries
	between their class and all others less often.

	Concretely, let F(x) be the vector of the model's non-activated outputs on
	input x of class j. Let

			L(x)_i = max(0, gamma + d * sign(F(x)_i - F(x)_j))

	where i ≠ j, gamma is the margin size, d is the approximated distance from x
	to the decision boundary between classes i and j

			d = |F(x)_i - F(x)_j| / ||F'(x)_i - F'(x)_j||

	and sign() returns positive if F(x) is incorrectly classified and negative
	otherwise. Suppose prior_reward is the prior reward for example x. Then let

			R(x) = lambda * (E_{i ≠ j} L(x)_i) + prior_reward * (1 - lambda)

	where lambda is a constant between zero and one giving how much to weight
	the newer data over past data. Then the updated reward for example x is

			new_reward = R(x) / number of times x has been sampled
	"""

	def __init__(self, dataset, num_batches, beta=1.414, decay=.975,
		gamma=10000.,
		alpha_factor=4.0,
		top_k=1,
		dist_norm=2,
		epsilon=1e-8,
		use_approximation=True,
		loss_type="all_top_k"):
		"""
		Args:
		dataset			-- the dataset to sample from
		num_examples	-- the number of batches to return per call to
							__iter__()
		beta			-- the constant by which to multiply the exploration
							term
		decay			-- the rate at which to decay old losses

		gamma (float): Desired margin, and distance to boundary above the margin
						will be clipped.
		alpha_factor (float): Factor to determine the lower bound of margin.
						Both gamma and alpha_factor determine points to include
						in training the margin these points lie with distance to
						boundary of [gamma * (1 - alpha), gamma]
		top_k (int):Number of top classes to include in the margin loss.
		dist_norm (1, 2, np.inf): Distance to boundary defined on norm
		epslion (float): Small number to avoid division by 0.
		use_approximation (bool):
		loss_type ("all_top_k", "worst_top_k", "avg_top_k"):  If 'worst_top_k'
						  only consider the minimum distance to boundary of the
						top_k classes. If 'average_top_k' consider average
						distance to boundary. If 'all_top_k' consider all top_k.
						When top_k = 1, these choices are equivalent.
		"""
		BanditSampler.__init__(self, dataset, num_batches, beta=1.414, decay=.6)
		self.dist_upper = gamma
		self.dist_lower = gamma * (1.0 - alpha_factor)

		self.alpha = alpha_factor
		self.top_k = top_k
		self.dual_norm = {1: np.inf, 2: 2, np.inf: 1}[dist_norm]
		self.eps = epsilon

		self.use_approximation = use_approximation
		self.loss_type = loss_type


		def get_reward_update(outputs, label, feature_maps):
			""" Returns the update to the reward of

			This code is derived from a PyTorch implementation of the original
			paper, at [https://github.com/zsef123/Large_Margin_Loss_PyTorch].
			Many thanks!

			Args:
			outputs			-- the model's output on the input being updated
			label			-- the correct label for the outputs
			feature_maps	-- the input, at any stage of being computed upon
			"""
			onehot_labels = torch.zeros(len(outputs))
			onehot_lables[int(label)] = 1.

			# CODE FROM [https://github.com/zsef123/Large_Margin_Loss_PyTorch/]#
			def _max_with_relu(a, b):
				return a + F.relu(b - a)

			def _get_grad(out_, in_):
				grad, *_ = torch.autograd.grad(out_, in_,
					grad_outputs=torch.ones_like(out_, dtype=torch.float32),
					retain_graph=True)
				return grad.view(in_.shape[0], -1)

			prob = F.softmax(logits, dim=1)
			correct_prob = prob * onehot_labels

			correct_prob = torch.sum(correct_prob, dim=1, keepdim=True)
			other_prob = prob * (1.0 - onehot_labels)

			if self.top_k > 1:
				topk_prob, _ = other_prob.topk(self.top_k, dim=1)
			else:
				topk_prob, _ = other_prob.max(dim=1, keepdim=True)

			diff_prob = correct_prob - topk_prob

			loss = torch.empty(0, device=logits.device)
			for feature_map in feature_maps:
				diff_grad = torch.stack(
					[_get_grad(diff_prob[:, i], feature_map) for i in range(self.top_k)],
					dim=1)
				diff_gradnorm = torch.norm(diff_grad, p=self.dual_norm, dim=2)

				if self.use_approximation:
					diff_gradnorm.detach_()

				dist_to_boundary = diff_prob / (diff_gradnorm + self.eps)

				if self.loss_type == "worst_top_k":
					dist_to_boundary, _ = dist_to_boundary.min(dim=1)
				elif self.loss_type == "avg_top_k":
					dist_to_boundary = dist_to_boundary.mean(dim=1)

				loss_layer = _max_with_relu(dist_to_boundary, self.dist_lower)
				loss_layer = _max_with_relu(0, self.dist_upper - loss_layer) - self.dist_upper
				loss = torch.cat([loss, loss_layer])
			return loss.mean()

		def update_sampler(self, outputs, labels, indices) -> None:
			"""Updates the sampler's data.

			Args:
			outputs	-- a tensor of the model's outputs
			labels	-- a tensor of labels for [inputs]
			indices	-- a tensor of indices for [inputs]
			"""
			self.total_times += len(indices)

			def fn(i, idx):
				return get_reward_update(outputs[i], labels[i], self.dataset[idx][0])
			bandit_losses = [fn(i, idx) for i,idx in enumerate(indices)]

			for i, loss in enumerate(bandit_losses):
				idx = indices[i]
				self.subset_to_loss[idx] = loss + self.subset_to_loss[index] * self.decay
				self.subset_to_times[idx] += 1

			explore = torch.sqrt(torch.log(self.subset_to_times) / self.total_times)
			reward = self.subset_to_loss / self.subset_to_times
			self.ucb_values = reward + self.beta * explore


################################################################################
###   OLD IDEA CODE    														   #
################################################################################

			# def compute_update(o, i):
			# 	""" [o] is a model's output on the [ith] example in [labels]"""
			# 	values, indices = torch.topk(o[i], 2)
			# 	if labels[i] in indices:
			# 		return values[0] - values[1]
			# 	else:
			# 		return values[0] - o[labels[i]]
			# 	return ((sum(o) - o[labels[i]]) / (self.num_classes - 1)) + 1 - o[labels[i]]
