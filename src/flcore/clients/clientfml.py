import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientavg import clientAvg
from flcore.clients.clientbase import save_item


class clientFML(clientAvg):
    """
    Federated Mutual Learning client.

    Each client keeps its heterogeneous local model and trains a homogeneous
    auxiliary global model through bidirectional distillation.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.alpha = float(getattr(args, "fml_alpha", 0.5))
        self.beta = float(getattr(args, "fml_beta", 0.5))
        self.temperature = float(getattr(args, "fml_temperature", 1.0))
        self.global_model = None

    def set_global_model(self, global_model):
        self.global_model = copy.deepcopy(global_model).to(self.device)

    def _kl_loss(self, student_logits, teacher_logits):
        temperature = max(self.temperature, 1e-6)
        return F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits.detach() / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

    def train(self):
        if self.global_model is None:
            raise RuntimeError("FML client requires a global auxiliary model before training.")

        trainloader = self.load_train_data()
        model = self.model
        global_model = self.global_model

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_global = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
        model.train()
        global_model.train()

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

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = model(x)
                output_global = global_model(x)

                local_loss = (
                    self.alpha * self.loss(output, y)
                    + (1.0 - self.alpha) * self._kl_loss(output, output_global)
                )
                global_loss = (
                    self.beta * self.loss(output_global, y)
                    + (1.0 - self.beta) * self._kl_loss(output_global, output)
                )

                optimizer.zero_grad()
                optimizer_global.zero_grad()
                local_loss.backward(retain_graph=True)
                global_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer_global.step()

        self.model = model
        self.global_model = global_model

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def save_best_model(self):
        self.best_model = copy.deepcopy(self.model)
        save_item(self.best_model, self.role, "best_model", self.save_folder_name)

