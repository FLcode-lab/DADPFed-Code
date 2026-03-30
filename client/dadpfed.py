import torch

from utils import *
from .client import Client


class dadpfed(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(dadpfed, self).__init__(device, model_func, received_vecs, dataset, lr, args)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.global_params_list = self.received_vecs['Global_params_list'].to(self.device)
        self.local_drift_list = self.received_vecs['Local_drift_list'].to(self.device)
        self.use_mask = bool(self.received_vecs.get('Use_mask', False))

        self.mask_with_shape = None
        if self.use_mask:
            self.mask_with_shape = get_params_list_with_shape(
                self.model,
                self.received_vecs['Mask_list'],
                self.device,
            )

    @torch.no_grad()
    def _apply_mask(self):
        if not self.use_mask or self.mask_with_shape is None:
            return
        for param, mask in zip(self.model.parameters(), self.mask_with_shape):
            param.mul_(mask)

    def _regularized_loss(self, predictions, labels):
        loss_pred = self.loss(predictions, labels)
        param_list = param_to_vector(self.model)
        drift_linear = torch.sum(param_list * self.local_drift_list)
        prox_term = 0.5 * abs(self.args.alpha) * torch.sum((param_list - self.global_params_list) ** 2)
        return loss_pred - drift_linear + prox_term

    def train(self):
        self.model.train()

        for _ in range(self.args.local_epochs):
            for inputs, labels in self.dataset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                predictions = self.model(inputs)
                loss = self._regularized_loss(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
                self._apply_mask()

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list
        self.comm_vecs['local_drift_list'] = (
            self.received_vecs['Local_drift_list']
            - self.args.alpha * (last_state_params_list - self.received_vecs['Global_params_list'])
        )

        return self.comm_vecs
