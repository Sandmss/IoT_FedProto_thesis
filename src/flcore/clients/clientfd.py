import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientavg import clientAvg
from flcore.clients.clientbase import save_item


class clientFD(clientAvg):
    """
    FedDistill client.

    The client uploads class-wise average logits and learns from the server's
    aggregated logits in the next communication round.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.lamda = args.lamda
        self.temperature = float(getattr(args, "fd_temperature", 1.0))
        self.global_logits = None
        self.local_logits = {}

    def _distill_loss(self, output, y):
        if not self.global_logits:
            return output.new_tensor(0.0)

        targets = []
        mask = []
        for label in y.detach().cpu().tolist():
            global_logit = self.global_logits.get(int(label))
            if isinstance(global_logit, torch.Tensor):
                targets.append(global_logit.detach().to(output.device, dtype=output.dtype))
                mask.append(True)
            else:
                targets.append(torch.zeros_like(output[0]))
                mask.append(False)

        if not any(mask):
            return output.new_tensor(0.0)

        target_logits = torch.stack(targets, dim=0)
        valid_mask = torch.tensor(mask, device=output.device, dtype=torch.bool)
        student = output[valid_mask]
        teacher = target_logits[valid_mask]
        temperature = max(self.temperature, 1e-6)

        return F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

    def train(self):
        trainloader = self.load_train_data()
        model = self.model

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max(2, max_local_epochs // 2 + 1)))

        logits = defaultdict(list)
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
                loss = self.loss(output, y)
                loss = loss + self.lamda * self._distill_loss(output, y)

                for row, label in zip(output.detach(), y.detach().cpu().tolist()):
                    logits[int(label)].append(row.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        self.model = model
        self.local_logits = agg_logits(logits)
        save_item(self.local_logits, self.role, "logits", self.save_folder_name)

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def save_best_model(self):
        self.best_model = copy.deepcopy(self.model)
        save_item(self.best_model, self.role, "best_model", self.save_folder_name)


def agg_logits(logits):
    averaged = {}
    for label, logit_list in logits.items():
        if len(logit_list) > 1:
            averaged[label] = torch.stack(logit_list, dim=0).mean(dim=0)
        else:
            averaged[label] = logit_list[0]
    return averaged

