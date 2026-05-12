import copy
import time

import torch

from flcore.clients.clientbase import save_item
from flcore.clients.clientlg import clientLGFedAvg
from flcore.servers.serverbase import Server


class LGFedAvg(Server):
    """
    Adapted LG-FedAvg server.

    This implementation follows the official local/global representation idea,
    but uses a heterogeneous-safe split for the current project:
    - local feature extractor (`base.*`) stays on each client
    - shared classifier head (`head.*`) is averaged across selected clients
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLGFedAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("LG-FedAvg server and clients are ready.")

        self.Budget = []
        self.best_test_acc = (0, 0)
        self.shared_param_prefixes = tuple(
            prefix.strip()
            for prefix in getattr(args, "lg_shared_param_prefixes", "head.").split(",")
            if prefix.strip()
        ) or ("head.",)
        self.global_shared_state = self._extract_shared_state(self.clients[0].model)
        save_item(self.global_shared_state, self.role, "global_shared_state", self.save_folder_name)

    def _extract_shared_state(self, model):
        shared_state = {}
        for name, tensor in model.state_dict().items():
            if any(name.startswith(prefix) for prefix in self.shared_param_prefixes):
                shared_state[name] = tensor.detach().clone()
        return shared_state

    def set_global_shared_to_clients(self, clients=None):
        target_clients = self.clients if clients is None else clients
        for client in target_clients:
            client.set_shared_state(self.global_shared_state)

    def aggregate_shared_parameters(self):
        assert len(self.selected_clients) > 0

        total_train_samples = sum(max(client.train_samples, 0) for client in self.selected_clients)
        aggregated_state = {
            name: torch.zeros_like(tensor)
            for name, tensor in self.global_shared_state.items()
        }

        for client in self.selected_clients:
            weight = (
                client.train_samples / total_train_samples
                if total_train_samples > 0
                else 1.0 / len(self.selected_clients)
            )
            client_shared = client.get_shared_state()
            for name, tensor in client_shared.items():
                if name in aggregated_state:
                    aggregated_state[name] += tensor.to(aggregated_state[name].device) * weight

        self.global_shared_state = {
            name: tensor.detach().clone()
            for name, tensor in aggregated_state.items()
        }
        save_item(self.global_shared_state, self.role, "global_shared_state", self.save_folder_name)

    def estimate_shared_comm_params(self):
        selected_clients = self.selected_clients if self.selected_clients else self.clients
        return float(sum(client.count_shared_params() for client in selected_clients))

    def train(self):
        print("\n------------- Global round: 0 (initial evaluation) -------------")
        self.set_global_shared_to_clients()
        self.evaluate()
        if self.rs_test_acc:
            self.best_test_acc = (self.rs_test_acc[-1], 0)
            print("Saving initial LG-FedAvg checkpoints at round 0...")
            self.save_best_checkpoint()
        print("----------------------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            stop_training = False

            self.selected_clients = self.select_clients()
            self.set_global_shared_to_clients(self.selected_clients)
            for client in self.selected_clients:
                client.train()

            self.aggregate_shared_parameters()
            self.set_global_shared_to_clients(self.selected_clients)
            self.record_round_overheads()

            if i % self.eval_gap == 0:
                print(f"\n------------- Global round: {i} (evaluation) -------------")
                self.set_global_shared_to_clients()
                self.evaluate()

                if self.rs_test_acc and self.rs_test_acc[-1] > self.best_test_acc[0]:
                    self.best_test_acc = (self.rs_test_acc[-1], i)
                    print(f"New best accuracy detected. Saving best checkpoint at round {i}...")
                    self.save_best_checkpoint()

                if self.auto_break and self.patience_should_stop_after_eval():
                    stop_training = True

            self.Budget.append(time.time() - s_t)
            print(f"Round time: {self.Budget[-1]:.2f}s")

            if stop_training:
                break

        print("\nTraining finished. Best accuracy summary:")
        if self.rs_test_acc:
            print(f"  Test accuracy: {self.best_test_acc[0]:.4f} (round {self.best_test_acc[1]})")

        if self.Budget:
            print(f"Average round time: {sum(self.Budget) / len(self.Budget):.2f}s")

        if getattr(self.args, "skip_figures", False):
            print("Skipping t-SNE generation (--skip_figures).")
        else:
            self.draw_tsne()
        self.save_results()

    def save_best_checkpoint(self):
        save_item(self.global_shared_state, self.role, "best_global_shared_state", self.save_folder_name)
        for client in self.clients:
            client.save_best_model()

    def draw_tsne(self):
        self.draw_feature_tsne(title="LG-FedAvg Feature t-SNE")

