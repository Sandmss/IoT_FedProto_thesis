import time
from collections import defaultdict

import torch

from flcore.clients.clientbase import save_item
from flcore.clients.clientfd import clientFD
from flcore.servers.serverbase import Server


class FD(Server):
    """
    FedDistill server.

    It aggregates class-wise logits from heterogeneous clients and broadcasts
    the resulting global logits for next-round local distillation.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("FD (FedDistill) server and clients are ready.")

        self.Budget = []
        self.best_test_acc = (0, 0)
        self.global_logits = None

    def set_global_logits_to_clients(self, clients=None):
        target_clients = self.clients if clients is None else clients
        for client in target_clients:
            client.global_logits = self.global_logits

    def receive_logits(self):
        assert len(self.selected_clients) > 0

        uploaded_logits = []
        for client in self.selected_clients:
            if client.local_logits:
                uploaded_logits.append(client.local_logits)

        if not uploaded_logits:
            print("Warning: no client logits were received in this round. Skipping FD server update.")
            return

        self.global_logits = aggregate_logits(uploaded_logits)
        save_item(self.global_logits, self.role, "global_logits", self.save_folder_name)

    def estimate_fd_comm_params(self):
        selected_clients = self.selected_clients if self.selected_clients else self.clients
        comm_params = 0
        for client in selected_clients:
            for logit in getattr(client, "local_logits", {}).values():
                if isinstance(logit, torch.Tensor):
                    comm_params += logit.numel()
        return float(comm_params)

    def train(self):
        print("\n------------- Global round: 0 (initial evaluation) -------------")
        self.evaluate()
        if self.rs_test_acc:
            self.best_test_acc = (self.rs_test_acc[-1], 0)
            print("Saving initial FD checkpoints at round 0...")
            self.save_best_checkpoint()
        print("----------------------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            stop_training = False

            self.selected_clients = self.select_clients()
            self.set_global_logits_to_clients(self.selected_clients)
            for client in self.selected_clients:
                client.train()

            self.receive_logits()
            self.record_round_overheads()

            if i % self.eval_gap == 0:
                print(f"\n------------- Global round: {i} (evaluation) -------------")
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
        if self.global_logits is not None:
            save_item(self.global_logits, self.role, "best_global_logits", self.save_folder_name)
        for client in self.clients:
            client.save_best_model()

    def draw_tsne(self):
        self.draw_feature_tsne(title="FD Feature t-SNE")


def aggregate_logits(local_logits_list):
    clusters = defaultdict(list)
    for local_logits in local_logits_list:
        for label, logit in local_logits.items():
            clusters[label].append(logit.detach().cpu())

    aggregated = {}
    for label, logit_list in clusters.items():
        if len(logit_list) > 1:
            aggregated[label] = torch.stack(logit_list, dim=0).mean(dim=0)
        else:
            aggregated[label] = logit_list[0]
    return aggregated

