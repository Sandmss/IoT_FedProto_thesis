# Output log file name: iot.out
python -u main.py -ab 1 -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 100 -nw 4 -nc 10 -nb 15 -dataset IoT -model_family IoT_MLP --input_dim 77 -fd 64 -did 0 -algo FedProto -lam 0.1 -se 100 -mart 100 > iot.out 2>&1
