# Output log file name: iot_local.out
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 100 -nw 4 -nc 10 -nb 15 -dataset IoT -model_family IoT_MLP --input_dim 77 -fd 512 -did 0 -algo Local --normal_class 0 -lam 0.1 -se 100 -mart 100 > iot_local.out 2>&1

