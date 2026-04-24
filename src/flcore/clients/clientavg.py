import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, save_item


class clientAvg(Client):
    """
    FedAvg client implementation.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

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

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = model(x)
                loss = self.loss(output, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

        self.model = model

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def save_best_model(self):
        self.best_model = copy.deepcopy(self.model)
        save_item(self.best_model, self.role, 'best_model', self.save_folder_name)

    def extract_features(self, max_samples=None):
        model = self.best_model if self.best_model is not None else self.model

        testloader = self.load_test_data()
        model.eval()

        features_list = []
        labels_list = []
        collected = 0
        rng = np.random.default_rng(0)

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = model.base(x)

                rep_np = rep.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                if max_samples is not None:
                    remaining = max_samples - collected
                    if remaining <= 0:
                        break
                    if len(rep_np) > remaining:
                        sample_idx = rng.choice(len(rep_np), size=remaining, replace=False)
                        rep_np = rep_np[sample_idx]
                        y_np = y_np[sample_idx]

                features_list.append(rep_np)
                labels_list.append(y_np)
                collected += len(rep_np)
                if max_samples is not None and collected >= max_samples:
                    break

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            return features, labels
        return np.array([]), np.array([])
