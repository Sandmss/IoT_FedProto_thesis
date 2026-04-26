import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from flcore.clients.clientbase import debug_log, load_item, save_item
from flcore.clients.clientproto import clientproto
from flcore.servers.serverbase import Server


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientproto)
        self.global_protos = None

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("FedProto server and clients are ready.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.best_test_acc = (0, 0)
        self.best_global_test_acc = (0, 0)
        self.best_local_proto_test_acc = (0, 0)
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

    def set_global_protos_to_clients(self, clients=None):
        target_clients = self.clients if clients is None else clients
        for client in target_clients:
            client.global_protos = self.global_protos

    def train(self):
        print("\n------------- Global round: 0 (initial evaluation) -------------")
        self.set_global_protos_to_clients()
        if self.global_protos is not None:
            self.evaluate()
        print("----------------------------------------------------------------")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            stop_training = False
            print(f"\n------------- Global round: {i} -------------")

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.current_round = i

            if i in {1, 10, 50, 98, 100}:
                debug_log(
                    "src/flcore/servers/serverproto.py:selected_clients",
                    "selected clients for round",
                    {
                        "round": i,
                        "selected_client_ids": [client.id for client in self.selected_clients],
                        "join_ratio": self.join_ratio,
                    },
                    run_id="fedproto-runtime",
                    hypothesis_id="H1",
                )

            self.set_global_protos_to_clients(self.selected_clients)
            for client in self.selected_clients:
                client.train()

            self.aggregate_protos()
            self.record_round_overheads()

            if i % self.eval_gap == 0:
                print(f"--- Round {i} evaluation ---")
                for client in self.clients:
                    client.current_round = i
                self.set_global_protos_to_clients()
                self.evaluate()

                if self.rs_test_acc and self.rs_test_acc[-1] > self.best_test_acc[0]:
                    self.best_test_acc = (self.rs_test_acc[-1], i)
                    print(f"New best accuracy detected. Saving best checkpoint at round {i}...")
                    self.save_best_checkpoint()

                if self.auto_break and self.patience_should_stop_after_eval():
                    stop_training = True

            if i % 50 == 0 and self.best_test_acc[1] > 0:
                print("\n--- Best accuracy so far ---")
                print(f"Local test set: {self.best_test_acc[0]:.4f} (round {self.best_test_acc[1]})")

            self.Budget.append(time.time() - s_t)
            print(f"Round time: {self.Budget[-1]:.2f}s")

            if stop_training:
                break

        print("\nTraining finished. Best accuracy summary:")
        if self.rs_test_acc:
            print(f"  Local test set: {max(self.rs_test_acc):.4f}")

        if self.Budget:
            print(f"Average round time: {sum(self.Budget) / len(self.Budget):.2f}s")
        if getattr(self.args, "skip_figures", False):
            print("Skipping t-SNE / prototype figures (--skip_figures).")
        else:
            self.draw_tsne()
            self.draw_proto_distribution_tsne()
        self.save_results()

    def aggregate_protos(self):
        assert len(self.selected_clients) > 0

        uploaded_protos_per_client = []
        uploaded_weights_per_client = []
        for client in self.selected_clients:
            protos = client.local_protos
            if protos:
                uploaded_protos_per_client.append(protos)
                uploaded_weights_per_client.append(client.local_proto_weights or {})

        if not uploaded_protos_per_client:
            print("Warning: no client prototypes were received in this round. Skipping server update.")
            return

        global_protos = proto_aggregation_with_weights(
            uploaded_protos_per_client,
            uploaded_weights_per_client,
        )

        self.global_protos = global_protos
        print(
            "Server aggregated prototypes with class-count weighting from "
            f"{len(uploaded_protos_per_client)} clients."
        )

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        existing_classes = list(global_protos.keys())
        class_counts = {
            int(class_id): int(sum(1 for protos in uploaded_protos_per_client if class_id in protos))
            for class_id in existing_classes
        }

        for k1 in existing_classes:
            for k2 in existing_classes:
                if k1 > k2:
                    dis = torch.norm(global_protos[k1] - global_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)

        self.min_gap = torch.min(self.gap)
        for idx in range(len(self.gap)):
            if self.gap[idx] > torch.tensor(1e8, device=self.device):
                self.gap[idx] = self.min_gap
        self.max_gap = torch.max(self.gap)

        print("\n--- Inter-class Margins ---")
        print(f"Per-class minimum gap: {self.gap.detach().cpu().numpy()}")
        print(f"Global minimum gap (Min Gap): {self.min_gap.item():.4f}")
        print(f"Global maximum gap (Max Gap): {self.max_gap.item():.4f}")
        print("---------------------------")

        current_round = getattr(self.selected_clients[0], "current_round", -1)
        if current_round in {1, 10, 50, 98, 100}:
            proto_norms = {
                int(class_id): float(torch.norm(proto).item())
                for class_id, proto in global_protos.items()
                if isinstance(proto, torch.Tensor)
            }
            debug_log(
                "src/flcore/servers/serverproto.py:aggregate_protos",
                "aggregated global prototypes summary",
                {
                    "round": current_round,
                    "uploaded_client_count": len(uploaded_protos_per_client),
                    "existing_classes": sorted(int(class_id) for class_id in existing_classes),
                    "class_counts": class_counts,
                    "min_gap": float(self.min_gap.item()),
                    "max_gap": float(self.max_gap.item()),
                    "proto_norms": proto_norms,
                },
                run_id="fedproto-runtime",
                hypothesis_id="H3",
            )

    def save_best_checkpoint(self):
        if self.global_protos is not None:
            save_item(self.global_protos, self.role, "best_global_protos", self.save_folder_name)
        for client in self.clients:
            client.save_best_model()

    def _get_figure_output_dir(self):
        return super()._get_figure_output_dir()

    def _build_figure_prefix(self):
        return super()._build_figure_prefix()

    def _load_item_if_exists(self, role, item_name):
        file_path = os.path.join(self.save_folder_name, f"{role}_{item_name}.pt")
        if not os.path.isfile(file_path):
            return None
        return load_item(role, item_name, self.save_folder_name)

    def draw_tsne(self):
        max_tsne_samples = 10000
        per_client_cap = max(1, max_tsne_samples // max(len(self.clients), 1))
        features_list = []
        labels_list = []

        for client in self.clients:
            if not hasattr(client, "extract_features"):
                continue
            features, labels = client.extract_features(max_samples=per_client_cap)
            if features.size == 0 or labels.size == 0:
                continue
            features_list.append(features)
            labels_list.append(labels)

        if not features_list:
            print("Skipping feature t-SNE: no client features available.")
            return

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        original_sample_count = len(features)
        if original_sample_count > max_tsne_samples:
            rng = np.random.default_rng(0)
            sample_indices = rng.choice(original_sample_count, size=max_tsne_samples, replace=False)
            features = features[sample_indices]
            labels = labels[sample_indices]
        if len(features) < 2:
            print("Skipping feature t-SNE: fewer than 2 samples.")
            return

        perplexity = min(30, len(features) - 1)
        if perplexity < 1:
            print("Skipping feature t-SNE: invalid perplexity.")
            return

        try:
            embedded = TSNE(
                n_components=2,
                random_state=0,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity,
            ).fit_transform(features)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                embedded[:, 0],
                embedded[:, 1],
                c=labels,
                cmap="tab20",
                s=10,
                alpha=0.8,
            )
            plt.title("FedProto Feature t-SNE")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(scatter, label="Class")
            plt.tight_layout()

            output_path = os.path.join(
                self._get_figure_output_dir(),
                f"{self._build_figure_prefix()}_feature_tsne.png",
            )
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"Saved feature t-SNE plot: {output_path}")
        except Exception as exc:
            plt.close("all")
            print(f"Skipping feature t-SNE: generation failed ({exc}).")

    def draw_proto_distribution_tsne(self):
        protos = self._load_item_if_exists(self.role, "best_global_protos")
        if protos is None:
            protos = self.global_protos

        if not protos:
            print("Skipping prototype visualization: no global prototypes available.")
            return

        class_ids = sorted(protos.keys())
        proto_vectors = []
        valid_class_ids = []
        for class_id in class_ids:
            proto = protos[class_id]
            if isinstance(proto, torch.Tensor):
                proto_vectors.append(proto.detach().cpu().numpy().reshape(-1))
                valid_class_ids.append(class_id)

        if len(proto_vectors) < 2:
            print("Skipping prototype visualization: fewer than 2 valid prototypes.")
            return

        proto_vectors = np.stack(proto_vectors, axis=0)

        try:
            if proto_vectors.shape[0] == 2:
                embedded = np.column_stack(
                    [proto_vectors[:, 0], np.zeros(proto_vectors.shape[0], dtype=np.float32)]
                )
            else:
                perplexity = min(5, proto_vectors.shape[0] - 1)
                if perplexity < 1:
                    print("Skipping prototype visualization: invalid perplexity.")
                    return
                embedded = TSNE(
                    n_components=2,
                    random_state=0,
                    init="pca",
                    learning_rate="auto",
                    perplexity=perplexity,
                ).fit_transform(proto_vectors)

            plt.figure(figsize=(8, 6))
            plt.scatter(embedded[:, 0], embedded[:, 1], s=80, c=valid_class_ids, cmap="tab20")
            for idx, class_id in enumerate(valid_class_ids):
                plt.annotate(str(class_id), (embedded[idx, 0], embedded[idx, 1]))
            plt.title("FedProto Global Prototype Distribution")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.tight_layout()

            output_path = os.path.join(
                self._get_figure_output_dir(),
                f"{self._build_figure_prefix()}_prototype_distribution.png",
            )
            plt.savefig(output_path, dpi=200)
            plt.close()
            print(f"Saved prototype distribution plot: {output_path}")
        except Exception as exc:
            plt.close("all")
            print(f"Skipping prototype visualization: generation failed ({exc}).")


def proto_aggregation_with_weights(protos_list, weights_list):
    """
    Aggregate client prototypes with class-specific sample-count weighting.
    """
    proto_clusters = defaultdict(list)
    weight_clusters = defaultdict(list)

    for protos, weights in zip(protos_list, weights_list):
        for key in protos.keys():
            proto_clusters[key].append(protos[key])
            weight_clusters[key].append(float(weights.get(key, 0)))

    aggregated_protos = defaultdict(list)
    for key in proto_clusters.keys():
        class_protos = torch.stack(proto_clusters[key])
        class_weights = torch.tensor(
            weight_clusters[key],
            dtype=class_protos.dtype,
            device=class_protos.device,
        ).view(-1, 1)
        total_weight = torch.sum(class_weights)

        if total_weight.item() <= 0:
            aggregated_protos[key] = torch.mean(class_protos, dim=0).detach()
        else:
            weighted_sum = torch.sum(class_protos * class_weights, dim=0)
            aggregated_protos[key] = (weighted_sum / total_weight).detach()

    return aggregated_protos
