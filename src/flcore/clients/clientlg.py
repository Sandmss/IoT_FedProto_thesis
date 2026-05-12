import copy
import time

import numpy as np
import torch

from flcore.clients.clientavg import clientAvg
from flcore.clients.clientbase import save_item


class clientLGFedAvg(clientAvg):
    """
    LG-FedAvg style client.

    In this project we adapt the original local/global split idea to the
    existing heterogeneous tabular models by only synchronizing the shared head
    parameters while keeping each client's feature extractor local.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.shared_param_prefixes = tuple(
            prefix.strip()
            for prefix in getattr(args, "lg_shared_param_prefixes", "head.").split(",")
            if prefix.strip()
        ) or ("head.",)

    def get_shared_state(self):
        shared_state = {}
        for name, tensor in self.model.state_dict().items():
            if any(name.startswith(prefix) for prefix in self.shared_param_prefixes):
                shared_state[name] = tensor.detach().clone()
        return shared_state

    def set_shared_state(self, shared_state):
        if not shared_state:
            return
        model_state = self.model.state_dict()
        for name, tensor in shared_state.items():
            if name in model_state and model_state[name].shape == tensor.shape:
                model_state[name] = tensor.detach().clone().to(model_state[name].device)
        self.model.load_state_dict(model_state, strict=False)

    def count_shared_params(self):
        return int(sum(tensor.numel() for tensor in self.get_shared_state().values()))

    def train(self):
        trainloader = self.load_train_data()
        model = self.model

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max(2, max_local_epochs // 2 + 1)))

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = model(x)
                loss = self.loss(output, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        self.model = model
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def save_best_model(self):
        self.best_model = copy.deepcopy(self.model)
        save_item(self.best_model, self.role, "best_model", self.save_folder_name)

