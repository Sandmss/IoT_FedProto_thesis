import time

from flcore.clients.clientlocal import clientLocal
from flcore.servers.serverbase import Server


class Local(Server):
    """
    Local baseline server wrapper.
    Clients train independently on their own data once and are then evaluated locally.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLocal)

        print(f"\nLocal baseline / 客户端总数: {self.num_clients}")
        print("Local 模式创建完成：仅执行客户端本地训练，不做服务器聚合。")

    def train(self):
        print("\n------------- Local Epoch: 0 (初始状态) -------------")
        self.evaluate()
        print("-----------------------------------------------------")

        start_time = time.time()

        for epoch in range(1, self.local_epochs + 1):
            print(f"\n------------- Local Epoch: {epoch} -------------")
            self.selected_clients = list(self.clients)
            for client in self.selected_clients:
                client.current_round = epoch
            for client in self.selected_clients:
                client.train(epochs=1)

            self.evaluate()

            if self.rs_test_acc:
                best_acc = max(self.rs_test_acc)
                if self.rs_test_acc[-1] >= best_acc:
                    for client in self.clients:
                        client.save_best_model()

        elapsed = time.time() - start_time
        print(f"Local baseline 总耗时: {elapsed:.2f} 秒")

        self.save_results()
