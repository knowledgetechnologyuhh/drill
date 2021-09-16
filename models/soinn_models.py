import logging

import torch
import torch.nn as nn

logger = logging.getLogger('SOINN-Log')


class ASOINN(nn.Module):
    def __init__(self, _init_size=2, **kwargs):
        super(ASOINN, self).__init__()
        self.size = _init_size
        self.pdist = nn.PairwiseDistance(2)

        # Hyperparameters for optimization
        self.eps_b = 1
        self.eps_n = 1

        # Thresholds
        self.t = self.register_buffer('t', torch.FloatTensor([1e10, 1e10]), )
        self.max_age = kwargs.get('max_age')
        self.del_freq = kwargs.get('del_freq')

        # Nodes and Edges
        self.V = self.register_buffer('V', torch.rand(self.size, kwargs.get('dim')), )
        self.E = None
        self.n = self.register_buffer('n', torch.zeros(self.size), )

    def register_buffer(self, name, tensor, **kwargs):
        super(ASOINN, self).register_buffer(name, tensor)
        return tensor

    def activate_bmus(self, sample):
        """
        Calculate pairwise distance of sample and network nodes, determine BMU and sBMU and update activity thresholds
        :param sample: Feature values of current observation
        :return: tuple of indices b (BMU) and s (sBMU)
        """
        dists = self.pdist(sample, self.V)
        return torch.topk(dists, k=2, largest=False, sorted=True)

    def update_threshold(self, idx):
        """
        Update activation threshold for BMU and sBMU given index of node in list of nodes V
        :param idx:
        :return: list of neighbors of current nodes
        """
        if self.n[idx] == 0:
            n_idx = None
            dists = self.pdist(self.V[idx], self.V)
            self.t[idx] = torch.topk(dists, k=2, largest=False, sorted=True).values[1]
        else:
            n_idx = self.E[1, (self.E[0] == idx)]
            dists = self.pdist(self.V[idx], self.V[n_idx])
            self.t[idx] = torch.topk(dists, k=1, largest=True, sorted=True).values
        return n_idx

    def insert_node(self, sample):
        """
        Insert new node into network at highest index in node list V, edge list E, and habituation list h.
        :param sample: Feature values of current observation
        """
        self.t = torch.cat((self.t, torch.FloatTensor([1e10]).to(self.n.device)))
        self.n = torch.cat((self.n, torch.zeros(1, device=self.n.device)))
        self.V = torch.cat((self.V, torch.unsqueeze(sample, 0)))
        self.size += 1

    def delete_nodes(self, pos):
        """
        Delete node without edge connections in node list V, edge list E, and habituation list h.
        :param pos: list of indices of nodes to be deleted
        """
        del_pos = torch.ones(self.n.shape, dtype=torch.bool, device=self.n.device)
        del_pos[pos] = False
        self.t = self.t[del_pos]
        self.n = self.n[del_pos]
        self.V = self.V[del_pos]
        for n in pos:
            self.E[0, (self.E[0] > n)] -= 1
            self.E[1, (self.E[1] > n)] -= 1
            self.size -= 1

    def update_bmu(self, sample, b, s, n_b):
        """
        Update BMU weight, create edge connection between BMU and sBMU and set connection age to 0
        :param sample: Feature values of current observation
        :param b: Index of BMU in 2D tensor V of network nodes
        :param s: Index of sBMU in 2D tensor V of network nodes
        :param n_b: Indices of BMU neighbors
        """
        self.V[b] += self.eps_b * (sample - self.V[b])

        # Case 1: No edges in network
        if self.E is None:
            self.E = torch.LongTensor([[b, s], [s, b], [1, 1]]).to(self.n.device)
            self.n[b] = 1
            self.n[s] = 1

        # Case 2: Edge between BMU and sBMU exists
        elif n_b is not None and s in n_b:
            self.E[2, ((self.E[0] == b) & (self.E[1] == s))] = 0
            self.E[2, ((self.E[0] == s) & (self.E[1] == b))] = 0

        # Case 3: No edge between BMU and sBMU
        else:
            self.E = torch.cat((self.E, torch.LongTensor([[b, s], [s, b], [1, 1]]).to(self.n.device)), dim=1)
            self.n[b] += 1
            self.n[s] += 1

    def update_neighbors(self, sample, b, n_b):
        """
        Update weights and emanating edges from BMU neighbors
        :param sample: Feature values of current observation
        :param b: Index of BMU in 2D tensor V of network nodes
        :param n_b: Indices of BMU neighbors
        """
        self.V[n_b] += self.eps_n * (sample - self.V[n_b])

        for n in n_b:
            e_pos = (((self.E[0] == b) | (self.E[1] == b)) & ((self.E[0] == n) | (self.E[1] == n)))

            # Increment age of all edges between BMU and neighbors by 1
            if self.E[2, e_pos][0] < self.max_age:
                self.E[2, e_pos] += 1

            # Remove edges that exceed age threshold
            else:
                self.E = self.E[:, ~e_pos]
                self.n[n] -= 1
                self.n[b] -= 1

    @torch.no_grad()
    def forward(self, it, data, **kwargs):
        """
        Adjusted SOINN algorithm as in Furao and Hasegawa. (2008), Section 3.1
        b: Index of BMU | s: Index of second BMU
        :param it: Number of batch/iteration
        :param data: List of mini-batch samples (contains just a single sample for continuous data stream)
        """
        y_pred = []
        for sample, label in data:

            # Activate BMU and sBMU (Step 3)
            (b_dist, s_dist), (b, s) = self.activate_bmus(sample)

            # TODO get label prediction via kNN
            y_pred.append(label.item())

            if self.training:

                # Update activity thresholds of BMU and sBMU (Step 3)
                n_b = self.update_threshold(b)
                self.update_threshold(s)

                # Node insertion criterion (Step 3)
                if b_dist > self.t[b] or s_dist > self.t[s]:

                    self.insert_node(sample)
                    logger.info('Iteration {}. Inserted new node at position (first dimensions): {}. '
                                'BMU: {}, sBMU: {}. Updated network size: {}.'.
                                format(it, sample.cpu().data.numpy()[:4].round(3), b, s, self.size))

                else:
                    # Create or reset connection between BMU and sBMU with zero age (Step 4)
                    self.update_bmu(sample, b, s, n_b)

                    if n_b is not None:
                        # Adapt weights and age edges of BMU neighbors (Step 5-7)
                        self.update_neighbors(sample, b, n_b)

                        # Delete BMU neighbors with no more emanating edges (Step 7)
                        delete = n_b[self.n[n_b] == 0]
                        if any(delete):
                            delete = torch.sort(delete, dim=0, descending=True).values
                            self.delete_nodes(delete)
                            logger.info('Iteration {}. Deleted node at indices: {}. Updated network size: {}.'.format(
                                it, delete, self.size))

                # Update learning rates (Step 6)
                self.eps_b = 1 / (it + 2)
                self.eps_n = 1 / ((it + 1) * 100)

        # Delete all nodes with at most one emanating edge (Step 8)
        if self.training and (it + 1) % self.del_freq == 0:
            delete0 = (self.n == 0).nonzero(as_tuple=True)[0]
            delete1 = (self.n == 1).nonzero(as_tuple=True)[0]
            for n in delete1:
                e_pos = ((self.E[0] == n) | (self.E[1] == n))
                self.n[self.E[0, e_pos]] -= 1
                self.E = self.E[:, ~e_pos]
            delete = torch.sort(torch.cat((delete0, delete1)), dim=0, descending=True).values
            self.delete_nodes(delete)
            logger.info('Iteration {}. Deleted node at indices: {}. Updated network size: {}.'.format(
                it, delete, self.size))

        return y_pred

class SOINNPLUS(ASOINN):
    def __init__(self, _init_size=3, pull_factor=100, cleanup_freq=1000, **kwargs):
        super().__init__(_init_size=_init_size, **kwargs)

        # Hyperparameters for Optimization
        self.pull_factor = pull_factor
        self.cleanup_freq = cleanup_freq
        self.delta = 1

        # Thresholds
        self.t = self.register_buffer('t', torch.ones(self.size), )

        # Node Attributes
        self.WT = self.register_buffer('WT', torch.ones(self.size), )
        self.IT = self.register_buffer('IT', torch.zeros(self.size), )
        self.T = self.register_buffer('T', torch.zeros(self.size), )

        # Label Matrix
        self.H = self.register_buffer('H', torch.zeros((self.size, 1)), )

        # Linked Winners Thresholds
        self.b_t = self.register_buffer('b_t', torch.tensor([]), )
        self.s_t = self.register_buffer('s_t', torch.tensor([]), )

        # Deleted Nodes/Edges Attributes
        self.A_del = self.register_buffer('A_del', torch.tensor([]), )
        self.U_del = self.register_buffer('U_del', torch.tensor([]), )

    def topk_bmus(self, sample, k=2):
        """
        Sort the V matrix by proximity to the BMU
        :param sample: Feature values of current observation
        :param k: Maximum number of indices to return
        :return b_idx: Indices of sorted neighbors
        """
        dists = self.pdist(sample, self.V)
        b_idx = torch.argsort(dists, descending=False)[:min(k, dists.size()[0])]
        return b_idx.tolist()

    def threshold_updating(self, idx):
        """
        Update activation threshold for BMU and sBMU given index of node in list of nodes V
        :param idx:
        :return: list of neighbors of current nodes
        """
        if self.n[idx] == 0:
            n_idx = torch.tensor([], device=self.n.device)
            dists = self.pdist(self.V[idx], self.V)
            self.t[idx] = torch.topk(dists, k=2, largest=False, sorted=True).values[1]
        else:
            n_idx = self.E[1, (self.E[0] == idx)]
            if not any(self.n[n_idx]) == 0:
                dists = self.pdist(self.V[idx], self.V[n_idx])
                self.t[idx] = torch.topk(dists, k=1, largest=True, sorted=True).values

        return n_idx

    def label_updating(self, idx, label):
        """
        Update label matrix H according to currently seen sample and corresponding label
        :param idx: Index of node in 2D tensor V of network nodes
        :param label: Class label of input observation
        """
        if self.H.shape[1] <= label:
            self.H = torch.cat((self.H, torch.zeros((self.size, label - self.H.shape[1] + 1), device=self.n.device)), dim=1)
        self.H[idx, label] += 1

    def node_adding(self, sample):
        """
        Insert new node into network at highest index in node list V, edge list E, and habituation list h.
        :param sample: Feature values of current observation
        """
        self.insert_node(sample)
        self.WT = torch.cat((self.WT, torch.ones(1, device=self.n.device)))
        self.IT = torch.cat((self.IT, torch.zeros(1, device=self.n.device)))
        self.T = torch.cat((self.T, torch.zeros(1, device=self.n.device)))
        self.H = torch.cat((self.H, torch.zeros((1, self.H.shape[1]), device=self.n.device)))

    def node_training(self, b, sample):
        """
        Adapt position of BMU
        :param b: Index of BMU in 2D tensor V of network nodes
        :param sample: Feature values of current observation
        """
        self.V[b] += self.eps_b * (sample - self.V[b])
        self.h[b] += self.tau_b * (1 - self.h[b]) - self.tau_b

    def node_merging(self, sample, b, n_b):
        self.WT[b] += 1
        self.V[b] += (sample - self.V[b])/self.WT[b]

        if self.n[b] > 0:
            self.V[n_b] += (sample - self.V[n_b])/(self.pull_factor * self.WT[n_b]).view(-1, 1)

        self.IT[b] = 0

    def node_linking(self, b, s, n_b):
        self.T = (self.WT - 1) / (torch.max(self.WT) - 1)

        if s not in n_b:
            b_std, b_mean = torch.std_mean(self.b_t)
            s_std, s_mean = torch.std_mean(self.s_t)

            if torch.sum(self.n) < 3 \
                    or self.t[b] * (1 - self.T[b]) < b_mean + 2 * b_std \
                    or self.t[s] * (1 - self.T[s]) < s_mean + 2 * s_std:
                if self.E is None:
                    self.E = torch.LongTensor([[b, s], [s, b], [1, 1]]).to(self.n.device)
                    self.n[b] = 1
                    self.n[s] = 1
                    self.b_t = self.t[b].view(-1)
                    self.s_t = self.t[s].view(-1)
                else:
                    self.E = torch.cat((self.E, torch.LongTensor([[b, s], [s, b], [1, 1]]).to(self.n.device)), dim=1)
                    self.n[b] += 1
                    self.n[s] += 1
                    self.b_t = torch.cat((self.b_t, self.t[b].view(-1)))
                    self.s_t = torch.cat((self.s_t, self.t[s].view(-1)))

        else:
            self.E[2, ((self.E[0] == b) & (self.E[1] == s))] = 0
            self.E[2, ((self.E[0] == s) & (self.E[1] == b))] = 0

        for n in n_b:
            self.E[2, (((self.E[0] == b) | (self.E[1] == b)) & ((self.E[0] == n) | (self.E[1] == n)))] += 1

    def edge_deletion(self, it, b):
        if self.n[b] == 0:
            return

        n_b, A = self.E[1:, (self.E[0] == b)]
        if len(A) == 0:
            return
        A25, A75 = torch.quantile(A.float(), torch.tensor([.25, .75], device=self.n.device))
        omega = A75 + 2 * (A75 - A25)
        n_del = len(self.A_del)

        edge_t = omega if n_del == 0 else \
            self.A_del.mean() * (n_del / (n_del + len(self.E[0]))) + omega * (1 - n_del / (n_del + len(self.E[0])))

        for n in n_b:
            e_pos = (((self.E[0] == b) | (self.E[1] == b)) & ((self.E[0] == n) | (self.E[1] == n)))
            e_age = 0 if len(self.E[2, e_pos]) == 0 else self.E[2, e_pos][0]
            if e_age > edge_t:
                self.A_del = e_age.float().view(-1) if n_del == 0 else torch.cat((self.A_del, e_age.view(-1)))
                self.E = self.E[:, ~e_pos]
                self.n[n] -= 1
                self.n[b] -= 1
                logger.info('Iteration {}. Deleted edge between BMU {} and neighbor {}.'.format(it, b, n))

    def node_deletion(self, it):

        if self.E is None or len(self.E[0]) == 0:
            return

        U = self.IT / self.WT
        u = U[(self.n > 0)]
        u_count = len(u)
        u_median = torch.quantile(u, .5)
        # Modified the deletion criteria. Uncomment sMAD to restore original functionality
        # u_upper = torch.quantile(u, .75)
        # sMAD = (u_upper - u_median)/3
        sMAD = 1.4826 * torch.quantile(torch.abs(u - u_median), .5)
        # end of modification
        omega = u_median + 2 * sMAD
        n_del = len(self.U_del)

        node_t = u_count / self.size if n_del == 0 else \
            self.U_del.mean() * (n_del / (n_del + u_count)) + omega * (1 - n_del / (n_del + u_count)) * (u_count / self.size)

        candidates = ((self.n == 0) & (U > node_t)).nonzero(as_tuple=False).view(-1)
        pos = torch.sort(candidates, descending=True).values
        if any(pos):
            del_pos = torch.ones(self.n.shape, dtype=torch.bool)
            del_pos[pos] = False
            self.t = self.t[del_pos]
            self.n = self.n[del_pos]
            self.V = self.V[del_pos]
            self.WT = self.WT[del_pos]
            self.IT = self.IT[del_pos]
            self.T = self.T[del_pos]
            self.H = self.H[del_pos]
            self.U_del = U[pos].view(-1) if len(self.U_del) == 0 else torch.cat((self.U_del, U[pos].view(-1)))
            for n in pos:
                self.E[0, (self.E[0] > n)] -= 1
                self.E[1, (self.E[1] > n)] -= 1
                self.size -= 1
            logger.info('Iteration {}. Deleted node at indices: {}. Updated network size: {}.'.format(
                it, pos, self.size))

    @torch.no_grad()
    def forward(self, it, data, n_bmus=3, supervised_sampling=False, **kwargs):
        """
        Adjusted SOINN algorithm as in Furao and Hasegawa. (2008), Section 3.1
        b: Index of BMU | s: Index of second BMU
        :param it: Number of batch/iteration
        :param data: List of mini-batch samples (contains just a single sample for continuous data stream)
        :n_bmus: N top nodes for each predicted class (or n closest neighbors)
        :supervised_sampling: Sample the highest nodes of a given class (otherwise samples closest neighbors)
        """
        y_pred = []
        v_bmus = []

        for sample, label in data:

            # Activate BMU and sBMU (Step 3)
            (b_dist, s_dist), (b, s) = self.activate_bmus(sample)

            y_pred.append(torch.argmax(self.H[b]).item())

            if self.training:

                # Update activity thresholds of BMU and sBMU (Step 3)
                n_b = self.threshold_updating(b)
                self.threshold_updating(s)

                # Node insertion criterion (Step 3)
                if b_dist <= self.t[b] or s_dist <= self.t[s]:

                    self.node_adding(sample)
                    self.label_updating(self.size - 1, label)
                    logger.info('Iteration {}. Inserted new node at position (first dimensions): {}. '
                                'BMU: {}, sBMU: {}. Updated network size: {}.'.
                                format(it, sample.cpu().data.numpy()[:4].round(3), b, s, self.size))

                else:
                    self.label_updating(b, label)
                    self.node_merging(sample, b, n_b)
                    self.node_linking(b, s, n_b)
                    self.edge_deletion(it, b)

                self.node_deletion(it)
                self.IT += 1

            if not supervised_sampling:
                v_bmu_idx = self.topk_bmus(sample, k=n_bmus)
                v_bmu_idx = v_bmu_idx if len(v_bmu_idx) == n_bmus else \
                    v_bmu_idx + ([v_bmu_idx[0]] * (n_bmus - len(v_bmu_idx)))
                v_bmus.append(self.V[v_bmu_idx].clone())

        # Node deletion phase (delete half of the nodes according to age)
        if self.training and (it + 1) % self.cleanup_freq == 0:
            pos = torch.sort(torch.argsort(self.IT, descending=True)[:(self.size // 4)], descending=True).values
            del_pos = torch.ones(self.n.shape, dtype=torch.bool)
            del_pos[pos] = False
            for n in pos:
                e_pos = ((self.E[0] == n) | (self.E[1] == n))
                self.n[self.E[0, e_pos]] -= 1
                self.E = self.E[:, ~e_pos]
                self.E[0, (self.E[0] > n)] -= 1
                self.E[1, (self.E[1] > n)] -= 1
                self.size -= 1
            self.t = self.t[del_pos]
            self.n = self.n[del_pos]
            self.V = self.V[del_pos]
            self.WT = self.WT[del_pos]
            self.IT = self.IT[del_pos]
            self.T = self.T[del_pos]
            self.H = self.H[del_pos]
            logger.info('Iteration {}. Deleted node at indices: {}. Updated network size: {}.'.format(
                it, pos, self.size))

        if supervised_sampling:
            # Get nodes with N most correct occurrences for a given prediction
            b_idx = torch.transpose(self.H, 0, 1)[y_pred]
            b_idx = torch.argsort(b_idx, descending=False, dim=1)
            b_idx = b_idx[:, :min(n_bmus, b_idx.size()[1])]
            v_bmus = self.V[b_idx]
        else:
            v_bmus = torch.stack(v_bmus)
        return y_pred, v_bmus
