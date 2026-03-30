import torch

from utils import *
from .dadpfed import dadpfed


class dadpfedsam(dadpfed):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(dadpfedsam, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        self.base_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.rho = self.args.rho

    def _grad_norm(self):
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm(p=2))
        if len(norms) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.norm(torch.stack(norms), p=2)

    def _forward_loss(self, inputs, labels):
        predictions = self.model(inputs)
        return self._regularized_loss(predictions, labels)

    def train(self):
        self.model.train()

        for _ in range(self.args.local_epochs):
            for inputs, labels in self.dataset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                self.base_optimizer.zero_grad()
                loss = self._forward_loss(inputs, labels)
                loss.backward()

                grad_norm = self._grad_norm()
                scale = self.rho / (grad_norm + 1e-12)
                perturbations = []
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is None:
                            perturbations.append(None)
                            continue
                        e_w = p.grad * scale
                        p.add_(e_w)
                        perturbations.append(e_w)

                self.base_optimizer.zero_grad()
                loss_second = self._forward_loss(inputs, labels)
                loss_second.backward()

                with torch.no_grad():
                    for p, e_w in zip(self.model.parameters(), perturbations):
                        if e_w is not None:
                            p.sub_(e_w)

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.base_optimizer.step()
                self._apply_mask()

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list
        self.comm_vecs['local_drift_list'] = (
            self.received_vecs['Local_drift_list']
            - self.args.alpha * (last_state_params_list - self.received_vecs['Global_params_list'])
        )

        return self.comm_vecs
