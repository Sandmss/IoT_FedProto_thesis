import copy
import time

from flcore.clients.clientavg import clientAvg
from flcore.clients.clientbase import save_item
from flcore.servers.serverbase import Server


class FedAvg(Server):
    """FedAvg server implementation."""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("FedAvg server and clients are ready.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.best_test_acc = (0, 0)

        client_model = self.clients[0].model
        self.global_model = copy.deepcopy(client_model).to(self.device)
        save_item(self.global_model, self.role, "global_model", self.save_folder_name)

    def set_global_model_to_clients(self, clients=None):
        target_clients = self.clients if clients is None else clients
        print("Syncing global model to clients for local evaluation/training...")
        for client in target_clients:
            client.model.load_state_dict(self.global_model.state_dict())

    def train(self):
        print("\n------------- Global round: 0 (initial evaluation) -------------")
        self.set_global_model_to_clients()
        self.evaluate()
        if self.rs_test_acc:
            self.best_test_acc = (self.rs_test_acc[-1], 0)
            print("Saving initial global checkpoint at round 0...")
            self.save_best_checkpoint()
        print("----------------------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            stop_training = False

            self.selected_clients = self.select_clients()
            self.set_global_model_to_clients(self.selected_clients)
            for client in self.selected_clients:
                client.train()

            self.aggregate_models()
            self.record_round_overheads()

            if i % self.eval_gap == 0:
                print(f"\n------------- Global round: {i} (evaluation) -------------")
                self.set_global_model_to_clients()
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

    def aggregate_models(self):
        assert len(self.selected_clients) > 0

        global_model = copy.deepcopy(self.global_model)
        for param in global_model.parameters():
            param.data.zero_()

        total_train_samples = sum(max(client.train_samples, 0) for client in self.selected_clients)

        for client in self.selected_clients:
            client_model = client.model
            weight = (
                client.train_samples / total_train_samples
                if total_train_samples > 0
                else 1.0 / len(self.selected_clients)
            )
            for gl_param, cl_param in zip(global_model.parameters(), client_model.parameters()):
                gl_param.data += cl_param.data * weight

        self.global_model = global_model

    def save_best_checkpoint(self):
        save_item(self.global_model, self.role, "best_global_model", self.save_folder_name)
        for client in self.clients:
            client.save_best_model()

    def draw_tsne(self):
        self.draw_feature_tsne(title="FedAvg Feature t-SNE")
