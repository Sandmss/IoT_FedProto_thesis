import numpy as np
import time
import torch

from flcore.clients.clientbase import Client, load_item, save_item, debug_log


class clientLocal(Client):
    """
    Local baseline: each client trains only on its own data.
    No server synchronization or aggregation is involved.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.current_round = 0

    def train(self, epochs=None):
        trainloader = self.load_train_data()
        model = self.model if self.model is not None else load_item(self.role, 'model', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs if epochs is None else epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max_local_epochs // 2 + 1))

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

        if self.id == 0 and self.current_round in {1, 10, 50, 99, 100}:
            debug_log(
                "src/flcore/clients/clientlocal.py:49",
                "local client train model state",
                {
                    "round": int(self.current_round),
                    "client_id": int(self.id),
                    "loaded_model_id": id(model),
                    "self_model_id_before_assign": id(self.model),
                    "loaded_model_norm": float(
                        sum(p.detach().float().norm().item() for p in model.parameters())
                    ),
                    "self_model_norm_before_assign": float(
                        sum(p.detach().float().norm().item() for p in self.model.parameters())
                    ) if self.model is not None else None,
                },
                run_id="local-runtime",
                hypothesis_id="L1",
            )

        self.model = model
        save_item(model, self.role, 'model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def save_best_model(self):
        model = self.model if self.model is not None else load_item(self.role, 'model', self.save_folder_name)
        save_item(model, self.role, 'best_model', self.save_folder_name)

    def extract_features(self, max_samples=None):
        model = self.model if self.model is not None else load_item(self.role, 'model', self.save_folder_name)
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

        if features_list:
            return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)
        return np.array([]), np.array([])
