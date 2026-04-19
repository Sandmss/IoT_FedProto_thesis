# Output log file name: iot_fedproto_20k_c20_noniid.out  (global_rounds=1000 with auto break)
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT_20k_c20_noniid -model_family IoT_MLP --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 0.1 -se 100 -mart 100 -ab True --skip_figures > iot_fedproto_20k_c20_noniid.out 2>&1

