import time
import numpy as np
import torch

from utils import *
from client import dadpfed
from .server import Server


class DADPFed(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(DADPFed, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)

        self.h_params_list = torch.zeros_like(init_par_list)
        print("     Global Drift (h)      --->  {:d}".format(self.h_params_list.shape[0]))

        self.local_drift_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print("      Local Drift (g)      --->  {:d} * {:d}".format(
            self.local_drift_params_list.shape[0], self.local_drift_params_list.shape[1]
        ))

        self.mask_list = torch.ones_like(init_par_list)
        self.use_mask = False

        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Global_params_list': init_par_list.clone().detach(),
            'Local_drift_list': torch.zeros_like(init_par_list),
            'Mask_list': torch.ones_like(init_par_list),
            'Use_mask': False,
        }
        self.Client = dadpfed

    def _build_drift_mask(self):
        eps = max(self.args.epsilon, 1e-12)
        q = torch.abs(self.h_params_list) / (torch.abs(self.server_model_params_list) + eps)
        quantile = float(np.clip(self.args.dadpfed_mask_quantile, 0.0, 1.0))
        tau = torch.quantile(q, quantile)
        return (q <= tau).float()

    def _build_importance_mask(self):
        retention = float(np.clip(self.args.dadpfed_retention, 1e-6, 1.0))
        num_params = self.server_model_params_list.numel()
        k = max(1, int(num_params * retention))
        if k >= num_params:
            return torch.ones_like(self.server_model_params_list)

        mask = torch.zeros_like(self.server_model_params_list)
        _, topk_idx = torch.topk(torch.abs(self.server_model_params_list), k=k, largest=True, sorted=False)
        mask[topk_idx] = 1.0
        return mask

    def _refresh_mask(self, t):
        cycle = max(1, int(self.args.dadpfed_cycle))
        if t % cycle != 0:
            self.use_mask = False
            self.mask_list = torch.ones_like(self.server_model_params_list)
            return

        drift_mask = self._build_drift_mask()
        importance_mask = self._build_importance_mask()
        combined_mask = drift_mask * importance_mask
        if torch.sum(combined_mask) == 0:
            combined_mask = importance_mask

        self.use_mask = True
        self.mask_list = combined_mask

    def _weighted_average_update(self, selected_clients):
        client_sizes = torch.tensor(
            [self.datasets.client_y[c].shape[0] for c in selected_clients],
            dtype=torch.float32,
        )
        weights = client_sizes / torch.sum(client_sizes)
        weighted_update = torch.zeros_like(self.server_model_params_list)
        for idx, client in enumerate(selected_clients):
            weighted_update += weights[idx] * self.clients_updated_params_list[client]
        return weighted_update

    def process_for_communication(self, client, masked_global_params):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(masked_global_params)
        else:
            self.comm_vecs['Params_list'].copy_(
                masked_global_params + self.args.beta * (self.server_model_params_list - self.clients_params_list[client])
            )
        self.comm_vecs['Global_params_list'].copy_(self.server_model_params_list)
        self.comm_vecs['Local_drift_list'].copy_(self.local_drift_params_list[client])
        self.comm_vecs['Mask_list'].copy_(self.mask_list)
        self.comm_vecs['Use_mask'] = self.use_mask

    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")

        for t in range(self.args.comm_rounds):
            start = time.time()
            selected_clients = self._activate_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            self._refresh_mask(t)
            masked_global_params = self.server_model_params_list * self.mask_list if self.use_mask \
                else self.server_model_params_list.clone()

            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, masked_global_params)
                _edge_device = self.Client(
                    device=self.device,
                    model_func=self.model_func,
                    received_vecs=self.comm_vecs,
                    dataset=dataset,
                    lr=self.lr,
                    args=self.args,
                )
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                if 'local_drift_list' in self.received_vecs:
                    self.local_drift_params_list[client] = self.received_vecs['local_drift_list']
                del _edge_device

            weighted_update = self._weighted_average_update(selected_clients)
            mean_gap = torch.mean(self.clients_params_list[selected_clients] - self.server_model_params_list, dim=0)
            self.h_params_list = self.h_params_list - self.args.alpha * mean_gap
            if self.use_mask:
                self.h_params_list = self.h_params_list * self.mask_list

            alpha_safe = self.args.alpha if abs(self.args.alpha) > 1e-12 else 1e-12
            self.server_model_params_list = (
                self.server_model_params_list
                + self.args.global_learning_rate * weighted_update
                - (1.0 / alpha_safe) * self.h_params_list
            )
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)

            self._test_(t, selected_clients)
            self._lr_scheduler_()

            end = time.time()
            self.time[t] = end - start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush=True)

        self._save_results_()
        self._summary_()
