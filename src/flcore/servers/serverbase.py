import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.model_family = args.model_family
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = int(getattr(args, "early_stop_patience", 100))
        self.auto_break = args.auto_break
        self.evals_since_improve = 0
        self.role = 'Server'
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_test_auc_micro = []
        self.rs_test_fnr = []
        self.rs_test_precision = []
        self.rs_test_recall = []
        self.rs_test_f1 = []
        self.rs_test_fpr = []
        self.rs_confusion_matrices = []
        self.rs_inference_latency_ms = []
        self.rs_train_loss = []
        self.rs_comm_params_per_round = []
        self.rs_comm_params_cumulative = []
        self.rs_model_params_mean = []
        self.rs_model_params_min = []
        self.rs_model_params_max = []
        self.rs_model_size_bytes_mean = []
        self.rs_model_size_bytes_min = []
        self.rs_model_size_bytes_max = []
        self.rs_model_flops_mean = []
        self.rs_model_flops_min = []
        self.rs_model_flops_max = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        global_model = load_item(client.role, 'model', client.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
    def save_results(self):
        algo = self._build_result_file_stem()
        result_path = self._get_metrics_output_dir()

        if (len(self.rs_test_acc)):
            file_path = os.path.join(result_path, f"{algo}.h5")
            print("Result file path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_test_auc_macro', data=self.rs_test_auc)
                hf.create_dataset('rs_test_auc_micro', data=self.rs_test_auc_micro)
                hf.create_dataset('rs_test_fnr', data=self.rs_test_fnr)
                hf.create_dataset('rs_test_precision', data=self.rs_test_precision)
                hf.create_dataset('rs_test_recall', data=self.rs_test_recall)
                hf.create_dataset('rs_test_f1', data=self.rs_test_f1)
                hf.create_dataset('rs_test_fpr', data=self.rs_test_fpr)
                hf.create_dataset('rs_confusion_matrices', data=self.rs_confusion_matrices)
                hf.create_dataset('rs_inference_latency_ms', data=self.rs_inference_latency_ms)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_comm_params_per_round', data=self.rs_comm_params_per_round)
                hf.create_dataset('rs_comm_params_cumulative', data=self.rs_comm_params_cumulative)
                hf.create_dataset('rs_model_params_mean', data=self.rs_model_params_mean)
                hf.create_dataset('rs_model_params_min', data=self.rs_model_params_min)
                hf.create_dataset('rs_model_params_max', data=self.rs_model_params_max)
                hf.create_dataset('rs_model_size_bytes_mean', data=self.rs_model_size_bytes_mean)
                hf.create_dataset('rs_model_size_bytes_min', data=self.rs_model_size_bytes_min)
                hf.create_dataset('rs_model_size_bytes_max', data=self.rs_model_size_bytes_max)
                hf.create_dataset('rs_model_flops_mean', data=self.rs_model_flops_mean)
                hf.create_dataset('rs_model_flops_min', data=self.rs_model_flops_min)
                hf.create_dataset('rs_model_flops_max', data=self.rs_model_flops_max)
        
        if 'temp' in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print("Temporary save directory deleted.")
            except:
                print("Temporary save directory was already removed.")

    def _get_results_root_dir(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
        )

    def _get_model_result_category(self):
        mapping = {
            "IoT_MLP": "MLP",
            "IoT_CNN1D": "CNN1D",
            "IoT_Transformer1D": "Transformer",
        }
        return mapping.get(self.model_family, "heterogeneous_models")

    def _get_algorithm_result_dir(self):
        result_dir = os.path.join(
            self._get_results_root_dir(),
            self._get_model_result_category(),
            self.algorithm,
        )
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def _get_metrics_output_dir(self):
        result_dir = os.path.join(self._get_algorithm_result_dir(), "metrics")
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def _get_figure_output_dir(self):
        result_dir = os.path.join(self._get_algorithm_result_dir(), "figures")
        os.makedirs(result_dir, exist_ok=True)
        return result_dir

    def _build_result_file_stem(self):
        return f"{self.dataset}_{self.algorithm}_{self.model_family}_{self.goal}_{self.times}"

    def _build_figure_prefix(self):
        return self._build_result_file_stem()

    def draw_feature_tsne(self, title=None):
        max_tsne_samples = 10000
        per_client_cap = max(1, max_tsne_samples // max(len(self.clients), 1))
        features_list = []
        labels_list = []
        rng = np.random.default_rng(0)

        for client in self.clients:
            if not hasattr(client, "extract_features"):
                continue
            try:
                features, labels = client.extract_features(max_samples=per_client_cap)
            except TypeError:
                features, labels = client.extract_features()
                if len(features) > per_client_cap:
                    sample_indices = rng.choice(len(features), size=per_client_cap, replace=False)
                    features = features[sample_indices]
                    labels = labels[sample_indices]
            if features.size == 0 or labels.size == 0:
                continue
            features_list.append(features)
            labels_list.append(labels)

        if not features_list:
            print("Skip feature t-SNE: no client features available.")
            return

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        if len(features) > max_tsne_samples:
            sample_indices = rng.choice(len(features), size=max_tsne_samples, replace=False)
            features = features[sample_indices]
            labels = labels[sample_indices]

        if len(features) < 2:
            print("Skip feature t-SNE: fewer than 2 samples.")
            return

        perplexity = min(30, len(features) - 1)
        if perplexity < 1:
            print("Skip feature t-SNE: invalid perplexity.")
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
            plt.title(title or f"{self.algorithm} Feature t-SNE")
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
            print(f"Skip feature t-SNE: generation failed ({exc}).")

    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc_macro = []
        tot_auc_micro = []
        tot_fnr = []
        tot_precision = []
        tot_recall = []
        tot_f1 = []
        tot_fpr = []
        confusion_matrices = []
        tot_latency_ms = []
        for c in self.clients:
            (
                ct,
                ns,
                auc_macro,
                auc_micro,
                fnr,
                precision,
                recall,
                f1,
                fpr,
                confusion_matrix,
                latency_ms,
            ) = c.test_metrics()
            tot_correct.append(ct*1.0)
            print(
                f'Client {c.id}: Acc: {ct*1.0/ns:.4f}, '
                f'AUC macro: {auc_macro:.4f}, AUC micro: {auc_micro:.4f}'
                f', Precision: {precision:.4f}, Recall: {recall:.4f}, '
                f'F1: {f1:.4f}, FNR: {fnr:.4f}, FPR: {fpr:.4f}, '
                f'Latency: {latency_ms:.4f} ms/sample '
                f'({int(ct)}/{ns} correct)'
            )
            tot_auc_macro.append(auc_macro * ns)
            tot_auc_micro.append(auc_micro * ns)
            tot_fnr.append(fnr)
            tot_precision.append(precision * ns)
            tot_recall.append(recall * ns)
            tot_f1.append(f1 * ns)
            tot_fpr.append(fpr)
            confusion_matrices.append(confusion_matrix)
            tot_latency_ms.append(latency_ms * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return (
            ids,
            num_samples,
            tot_correct,
            tot_auc_macro,
            tot_auc_micro,
            tot_fnr,
            tot_precision,
            tot_recall,
            tot_f1,
            tot_fpr,
            confusion_matrices,
            tot_latency_ms,
        )

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc_macro = sum(stats[3]) * 1.0 / sum(stats[1])
        test_auc_micro = sum(stats[4]) * 1.0 / sum(stats[1])
        test_fnr = float(np.mean(stats[5])) if len(stats[5]) > 0 else 0.0
        test_precision = sum(stats[6]) * 1.0 / sum(stats[1])
        test_recall = sum(stats[7]) * 1.0 / sum(stats[1])
        test_f1 = sum(stats[8]) * 1.0 / sum(stats[1])
        test_fpr = float(np.mean(stats[9])) if len(stats[9]) > 0 else 0.0
        test_confusion_matrix = np.sum(np.asarray(stats[10]), axis=0)
        test_latency_ms = sum(stats[11]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs_macro = [a / n for a, n in zip(stats[3], stats[1])]
        aucs_micro = [a / n for a, n in zip(stats[4], stats[1])]
        fnrs = stats[5]
        precisions = [a / n for a, n in zip(stats[6], stats[1])]
        recalls = [a / n for a, n in zip(stats[7], stats[1])]
        f1s = [a / n for a, n in zip(stats[8], stats[1])]
        fprs = stats[9]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_auc.append(test_auc_macro)
            self.rs_test_auc_micro.append(test_auc_micro)
            self.rs_test_fnr.append(test_fnr)
            self.rs_test_precision.append(test_precision)
            self.rs_test_recall.append(test_recall)
            self.rs_test_f1.append(test_f1)
            self.rs_test_fpr.append(test_fpr)
            self.rs_confusion_matrices.append(test_confusion_matrix)
            self.rs_inference_latency_ms.append(test_latency_ms)
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC Macro: {:.4f}".format(test_auc_macro))
        print("Averaged Test AUC Micro: {:.4f}".format(test_auc_micro))
        print("Averaged Test Precision Macro: {:.4f}".format(test_precision))
        print("Averaged Test Recall Macro: {:.4f}".format(test_recall))
        print("Averaged Test F1 Macro: {:.4f}".format(test_f1))
        print("Averaged Test FNR: {:.4f}".format(test_fnr))
        print("Averaged Test FPR: {:.4f}".format(test_fpr))
        print("Averaged Inference Latency: {:.4f} ms/sample".format(test_latency_ms))
        print("Global confusion matrix:")
        print(test_confusion_matrix)
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC Macro: {:.4f}".format(np.std(aucs_macro)))
        print("Std Test AUC Micro: {:.4f}".format(np.std(aucs_micro)))
        print("Std Test Precision Macro: {:.4f}".format(np.std(precisions)))
        print("Std Test Recall Macro: {:.4f}".format(np.std(recalls)))
        print("Std Test F1 Macro: {:.4f}".format(np.std(f1s)))
        print("Std Test FNR: {:.4f}".format(np.std(fnrs)))
        print("Std Test FPR: {:.4f}".format(np.std(fprs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def count_model_params(self, model):
        return sum(param.numel() for param in model.parameters())

    def estimate_model_param_stats(self):
        counts = []
        for client in self.clients:
            model = getattr(client, "model", None)
            if model is not None:
                counts.append(self.count_model_params(model))
        if not counts:
            return 0.0, 0.0, 0.0
        counts = np.asarray(counts, dtype=np.float64)
        return float(np.mean(counts)), float(np.min(counts)), float(np.max(counts))

    def estimate_model_size_bytes(self, model):
        total = 0
        for tensor in list(model.parameters()) + list(model.buffers()):
            total += tensor.numel() * tensor.element_size()
        return total

    def estimate_model_flops(self, model):
        """
        Estimate single-sample forward FLOPs for Linear and Conv1d layers.

        This lightweight estimator intentionally avoids extra dependencies. It
        covers the MLP/CNN paths exactly for Linear/Conv1d operations and gives
        a conservative lower-bound estimate for Transformer models because
        attention matmul internals are not exposed as simple modules.
        """
        was_training = model.training
        model.eval()
        flops = {"value": 0.0}
        handles = []

        def linear_hook(module, inputs, output):
            x = inputs[0]
            batch = x.shape[0] if x.dim() > 0 else 1
            output_elements = output.numel()
            flops["value"] += output_elements * module.in_features
            if module.bias is not None:
                flops["value"] += output_elements
            # Keep batch scaling explicit for unusual broadcasting inputs.
            if output.dim() == 1:
                flops["value"] *= batch

        def conv1d_hook(module, inputs, output):
            out = output
            batch = out.shape[0]
            out_channels = out.shape[1]
            out_length = out.shape[2]
            kernel_ops = module.kernel_size[0] * (module.in_channels / module.groups)
            bias_ops = 1 if module.bias is not None else 0
            flops["value"] += batch * out_channels * out_length * (kernel_ops + bias_ops)

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, torch.nn.Conv1d):
                handles.append(module.register_forward_hook(conv1d_hook))

        try:
            input_dim = int(getattr(self.args, "input_dim", 77))
            dummy = torch.zeros(1, input_dim, device=self.device)
            with torch.no_grad():
                model(dummy)
        except Exception:
            flops["value"] = 0.0
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                model.train()

        return float(flops["value"])

    def estimate_model_efficiency_stats(self):
        sizes = []
        flops = []
        for client in self.clients:
            model = getattr(client, "model", None)
            if model is not None:
                sizes.append(self.estimate_model_size_bytes(model))
                flops.append(self.estimate_model_flops(model))

        if not sizes:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        sizes = np.asarray(sizes, dtype=np.float64)
        flops = np.asarray(flops, dtype=np.float64)
        return (
            (float(np.mean(sizes)), float(np.min(sizes)), float(np.max(sizes))),
            (float(np.mean(flops)), float(np.min(flops)), float(np.max(flops))),
        )

    def estimate_round_comm_params(self):
        """
        Estimate per-round communication using the FedProto paper's Table 1 style.

        FedAvg-style methods count uploaded model parameters from selected clients.
        FedProto counts uploaded local class prototypes from selected clients.
        Local has no network communication.
        """
        if self.algorithm == "Local":
            return 0.0

        selected_clients = self.selected_clients if self.selected_clients else self.clients
        if self.algorithm == "FedProto":
            comm_params = 0
            for client in selected_clients:
                local_protos = getattr(client, "local_protos", None) or {}
                for proto in local_protos.values():
                    if isinstance(proto, torch.Tensor):
                        comm_params += proto.numel()
            return float(comm_params)

        return float(
            sum(
                self.count_model_params(client.model)
                for client in selected_clients
                if getattr(client, "model", None) is not None
            )
        )

    def record_round_overheads(self):
        comm_params = self.estimate_round_comm_params()
        previous_total = self.rs_comm_params_cumulative[-1] if self.rs_comm_params_cumulative else 0.0
        model_mean, model_min, model_max = self.estimate_model_param_stats()
        size_stats, flops_stats = self.estimate_model_efficiency_stats()

        self.rs_comm_params_per_round.append(comm_params)
        self.rs_comm_params_cumulative.append(previous_total + comm_params)
        self.rs_model_params_mean.append(model_mean)
        self.rs_model_params_min.append(model_min)
        self.rs_model_params_max.append(model_max)
        self.rs_model_size_bytes_mean.append(size_stats[0])
        self.rs_model_size_bytes_min.append(size_stats[1])
        self.rs_model_size_bytes_max.append(size_stats[2])
        self.rs_model_flops_mean.append(flops_stats[0])
        self.rs_model_flops_min.append(flops_stats[1])
        self.rs_model_flops_max.append(flops_stats[2])

        print(f"Communication params this round (paper-style): {comm_params:.0f}")
        print(f"Cumulative communication params: {previous_total + comm_params:.0f}")
        print(
            "Client model params mean/min/max: "
            f"{model_mean:.0f}/{model_min:.0f}/{model_max:.0f}"
        )
        print(
            "Client model size bytes mean/min/max: "
            f"{size_stats[0]:.0f}/{size_stats[1]:.0f}/{size_stats[2]:.0f}"
        )
        print(
            "Estimated forward FLOPs mean/min/max: "
            f"{flops_stats[0]:.0f}/{flops_stats[1]:.0f}/{flops_stats[2]:.0f}"
        )

    def patience_should_stop_after_eval(self):
        """
        Call once immediately after evaluate() appends a new global averaged test accuracy.

        Stops when there have been top_cnt consecutive evaluations where the new accuracy
        is not strictly greater than the best seen in all prior evaluations. This matches
        鈥渂est涔嬪悗杩炵画 N 娆¤瘎浼版病鏈夋洿楂樷€?semantics; the old check_done(top_cnt) logic did
        not work when the running argmax stayed at the last list index (e.g. monotonic acc).
        """
        if not self.auto_break:
            return False
        acc_ls = self.rs_test_acc
        if len(acc_ls) < 2:
            return False
        current = acc_ls[-1]
        past_best = max(acc_ls[:-1])
        if current > past_best:
            self.evals_since_improve = 0
        else:
            self.evals_since_improve += 1
        if self.evals_since_improve >= self.top_cnt:
            print(
                "Early stop: averaged test accuracy did not exceed historical best for "
                f"{self.top_cnt} consecutive evaluations."
            )
            return True
        return False

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


