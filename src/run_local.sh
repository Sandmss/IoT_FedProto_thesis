# Output log file name: iot_local_20k_c20_noniid.out  (local_epochs=1000 with auto break)
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1000 -nw 4 -nc 20 -nb 15 -dataset IoT_20k_c20_noniid -model_family IoT_MLP --input_dim 77 -fd 512 -did 0 -algo Local --normal_class 0 -lam 0.1 -se 100 -mart 100 -ab True > iot_local_20k_c20_noniid.out 2>&1

