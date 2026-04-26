# Experiment Summary

- Total summarized `.h5` files: `10`

| Model | Algorithm | Setting | Acc | AUC Macro | F1 | FNR | FPR | ResultFile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IoT_CNN1D | FedAvg | standard | 0.8936 | 0.995784 | 0.595766 | 0.0 | 0.0 | CNN1D/FedAvg/metrics/IoT_FedAvg_IoT_CNN1D_test_0.h5 |
| IoT_CNN1D | FedProto | standard | 0.944 | 0.99212 | 0.943591 | 0.0 | 0.13 | CNN1D/FedProto/metrics/IoT_FedProto_IoT_CNN1D_test_0.h5 |
| IoT_CNN1D | Local | standard | 0.9416 | 0.99146 | 0.941155 | 0.0 | 0.134 | CNN1D/Local/metrics/IoT_Local_IoT_CNN1D_test_0.h5 |
| IoT_MLP | FedAvg | standard | 0.868 | 0.991744 | 0.552969 | 0.0 | 0.0 | MLP/FedAvg/metrics/IoT_FedAvg_IoT_MLP_test_0.h5 |
| IoT_MLP | FedProto | standard | 0.9412 | 0.99176 | 0.940428 | 0.0 | 0.142 | MLP/FedProto/metrics/IoT_FedProto_IoT_MLP_test_0.h5 |
| IoT_MLP | Local | standard | 0.9392 | 0.991744 | 0.938437 | 0.0 | 0.148 | MLP/Local/metrics/IoT_Local_IoT_MLP_test_0.h5 |
| IoT_Transformer1D | FedAvg | standard | 0.8724 | 0.992248 | 0.578182 | 0.0 | 0.0 | Transformer/FedAvg/metrics/IoT_FedAvg_IoT_Transformer1D_test_0.h5 |
| IoT_Transformer1D | FedProto | standard | 0.9292 | 0.987756 | 0.928313 | 0.011 | 0.144 | Transformer/FedProto/metrics/IoT_FedProto_IoT_Transformer1D_test_0.h5 |
| IoT_Transformer1D | Local | standard | 0.9292 | 0.985728 | 0.928018 | 0.0 | 0.148 | Transformer/Local/metrics/IoT_Local_IoT_Transformer1D_test_0.h5 |
| IoT_MIX_MLP_CNN1D | FedProto | standard | 0.9416 | 0.992064 | 0.941107 | 0.0 | 0.126 | heterogeneous_models/FedProto/metrics/IoT_FedProto_IoT_MIX_MLP_CNN1D_test_0.h5 |
