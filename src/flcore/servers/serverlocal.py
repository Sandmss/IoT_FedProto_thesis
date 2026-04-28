import time

from flcore.clients.clientlocal import clientLocal
from flcore.servers.serverbase import Server


class Local(Server):
    """
    Local baseline server wrapper.
    Clients train independently on their own data and are evaluated locally.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLocal)
        self.best_test_acc = (0, 0)

        print(f"\nLocal baseline / total clients: {self.num_clients}")
        print("Local mode is ready: clients train independently without server aggregation.")

    def train(self):
        print("\n------------- Local epoch: 0 (initial evaluation) -------------")
        self.evaluate()
        if self.rs_test_acc:
            self.best_test_acc = (self.rs_test_acc[-1], 0)
            print("Saving initial local checkpoints at epoch 0...")
            for client in self.clients:
                client.save_best_model()
        print("---------------------------------------------------------------")

        start_time = time.time()

        for epoch in range(1, self.local_epochs + 1):
            print(f"\n------------- Local epoch: {epoch} -------------")
            self.selected_clients = list(self.clients)
            for client in self.selected_clients:
                client.current_round = epoch
            for client in self.selected_clients:
                client.train(epochs=1)

            self.record_round_overheads()
            self.evaluate()

            if self.rs_test_acc and self.rs_test_acc[-1] > self.best_test_acc[0]:
                self.best_test_acc = (self.rs_test_acc[-1], epoch)
                print(f"New best accuracy detected. Saving best local checkpoints at epoch {epoch}...")
                for client in self.clients:
                    client.save_best_model()

            if self.auto_break and self.patience_should_stop_after_eval():
                break

        elapsed = time.time() - start_time
        print("\nTraining finished. Best accuracy summary:")
        if self.rs_test_acc:
            print(f"  Test accuracy: {self.best_test_acc[0]:.4f} (epoch {self.best_test_acc[1]})")
        print(f"Local baseline total time: {elapsed:.2f}s")

        if getattr(self.args, "skip_figures", False):
            print("Skipping t-SNE generation (--skip_figures).")
        else:
            self.draw_feature_tsne(title="Local Feature t-SNE")
        self.save_results()
