# 读取多次实验结果，并汇总测试准确率 / AUC 的最佳值统计。
import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    results = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    best_acc = [run_result["rs_test_acc"].max() for run_result in results]
    print("std for best accurancy:", np.std(best_acc))
    print("mean for best accurancy:", np.mean(best_acc))

    if any(run_result["rs_test_auc_macro"].size > 0 for run_result in results):
        best_auc_macro = [run_result["rs_test_auc_macro"].max() for run_result in results]
        print("std for best auc macro:", np.std(best_auc_macro))
        print("mean for best auc macro:", np.mean(best_auc_macro))

    if any(run_result["rs_test_auc_micro"].size > 0 for run_result in results):
        best_auc_micro = [run_result["rs_test_auc_micro"].max() for run_result in results]
        print("std for best auc micro:", np.std(best_auc_micro))
        print("mean for best auc micro:", np.mean(best_auc_micro))

    for metric_name, label, reducer in [
        ("rs_test_precision", "best precision macro", np.max),
        ("rs_test_recall", "best recall macro", np.max),
        ("rs_test_f1", "best f1 macro", np.max),
        ("rs_test_fpr", "lowest fpr", np.min),
        ("rs_test_fnr", "lowest fnr", np.min),
        ("rs_inference_latency_ms", "inference latency ms/sample", np.mean),
        ("rs_comm_params_per_round", "communication params per round", np.mean),
        ("rs_comm_params_cumulative", "cumulative communication params", np.max),
        ("rs_model_params_mean", "mean model params", np.mean),
        ("rs_model_size_bytes_mean", "mean model size bytes", np.mean),
        ("rs_model_flops_mean", "mean estimated flops", np.mean),
    ]:
        if any(run_result[metric_name].size > 0 for run_result in results):
            values = [reducer(run_result[metric_name]) for run_result in results]
            print(f"std for {label}:", np.std(values))
            print(f"mean for {label}:", np.mean(values))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    all_results = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        all_results.append(read_data_then_delete(file_name, delete=False))

    return all_results


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))
        rs_test_auc_macro = _read_optional_dataset(hf, 'rs_test_auc_macro')
        if rs_test_auc_macro.size == 0:
            rs_test_auc_macro = _read_optional_dataset(hf, 'rs_test_auc')
        rs_test_auc_micro = _read_optional_dataset(hf, 'rs_test_auc_micro')
        rs_test_precision = _read_optional_dataset(hf, 'rs_test_precision')
        rs_test_recall = _read_optional_dataset(hf, 'rs_test_recall')
        rs_test_f1 = _read_optional_dataset(hf, 'rs_test_f1')
        rs_test_fpr = _read_optional_dataset(hf, 'rs_test_fpr')
        rs_test_fnr = _read_optional_dataset(hf, 'rs_test_fnr')
        rs_confusion_matrices = _read_optional_dataset(hf, 'rs_confusion_matrices')
        rs_inference_latency_ms = _read_optional_dataset(hf, 'rs_inference_latency_ms')
        rs_comm_params_per_round = _read_optional_dataset(hf, 'rs_comm_params_per_round')
        rs_comm_params_cumulative = _read_optional_dataset(hf, 'rs_comm_params_cumulative')
        rs_model_params_mean = _read_optional_dataset(hf, 'rs_model_params_mean')
        rs_model_size_bytes_mean = _read_optional_dataset(hf, 'rs_model_size_bytes_mean')
        rs_model_flops_mean = _read_optional_dataset(hf, 'rs_model_flops_mean')

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return {
        "rs_test_acc": rs_test_acc,
        "rs_test_auc_macro": rs_test_auc_macro,
        "rs_test_auc_micro": rs_test_auc_micro,
        "rs_test_precision": rs_test_precision,
        "rs_test_recall": rs_test_recall,
        "rs_test_f1": rs_test_f1,
        "rs_test_fpr": rs_test_fpr,
        "rs_test_fnr": rs_test_fnr,
        "rs_confusion_matrices": rs_confusion_matrices,
        "rs_inference_latency_ms": rs_inference_latency_ms,
        "rs_comm_params_per_round": rs_comm_params_per_round,
        "rs_comm_params_cumulative": rs_comm_params_cumulative,
        "rs_model_params_mean": rs_model_params_mean,
        "rs_model_size_bytes_mean": rs_model_size_bytes_mean,
        "rs_model_flops_mean": rs_model_flops_mean,
    }


def _read_optional_dataset(hf, dataset_name):
    dataset = hf.get(dataset_name)
    if dataset is None:
        return np.array([])
    return np.array(dataset)
