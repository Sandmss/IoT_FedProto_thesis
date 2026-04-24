# 鍩轰簬杞婚噺鍖栬仈閭﹀涔犵殑鐗╄仈缃戞伓鎰忔祦閲忔娴嬬郴缁?
## 椤圭洰绠€浠?鏈」鐩疄鐜颁簡涓€涓潰鍚?IoT 鎭舵剰娴侀噺妫€娴嬪満鏅殑杞婚噺鍖栬仈閭﹀涔犲師鍨嬬郴缁熴€? 
绯荤粺浠?`FedProto` 涓烘牳蹇冭仈閭︽柟妗堬紝鍚屾椂淇濈暀 `Local` 鍜?`FedAvg` 浣滀负瀵圭収鏂规硶锛屾敮鎸佸绉嶈交閲忓寲鏈湴妯″瀷锛屽苟鏀寔鍚屾瀯涓庨儴鍒嗗紓鏋勫疄楠屻€?
褰撳墠绯荤粺閲嶇偣鍖呮嫭锛?
- 杞婚噺鍖栨伓鎰忔祦閲忔娴嬫ā鍨?- 鑱旈偊璁粌娴佺▼瀹炵幇
- 澶氭寚鏍囩粨鏋滆瘎浼?- 缁撴灉褰掓。涓庢€荤粨鏋滆〃姹囨€?
## 褰撳墠鏀寔鍐呭

### 绠楁硶

- `Local`
- `FedAvg`
- `FedProto`

### 妯″瀷

- `IoT_MLP`
- `IoT_CNN1D`
- `IoT_Transformer1D`
- `IoT_MIX_MLP_CNN1D`
- `IoT_MIX_MLP_CNN_TRANS`

璇存槑锛?
- `Local / FedAvg / FedProto` 閫傜敤浜庡悓鏋勬ā鍨嬪疄楠?- 寮傛瀯妯″瀷瀹為獙褰撳墠浼樺厛寤鸿浣跨敤 `FedProto`
- 褰撳墠瀹炵幇涓嬩笉寤鸿鐩存帴灏嗗紓鏋勬ā鍨嬬敤浜?`FedAvg`

## 椤圭洰鐩綍

- `src`
  鏈€缁堜富浠ｇ爜鐩綍锛屽寘鍚缁冨叆鍙ｃ€佸鎴风銆佹湇鍔＄銆佹ā鍨嬪疄鐜颁笌缁撴灉姹囨€昏剼鏈?- `dataset`
  鑱旈偊鍒掑垎鍚庣殑瀹㈡埛绔暟鎹洰褰?- `data`
  鍘熷鏁版嵁涓庡鐞嗕腑闂翠骇鐗?- `results`
  瀹為獙鏃ュ織銆佹寚鏍囩粨鏋溿€佸浘鍍忚緭鍑哄拰姹囨€昏〃
- `reference`
  寮€棰樻姤鍛娿€佷换鍔′功銆佽鍒掓枃妗ｇ瓑鍙傝€冩潗鏂?- `sysytem`
  鏃х増鏈垨鍙傝€冪洰褰曪紝涓嶄綔涓烘渶缁堜富鍏ュ彛

## 鐜渚濊禆

寤鸿 Python 鐗堟湰锛?
- `Python 3.10+`

涓昏渚濊禆锛?
- `torch`
- `numpy`
- `scikit-learn`
- `h5py`
- `matplotlib`

瀹夎绀轰緥锛?
```bash
pip install torch numpy scikit-learn h5py matplotlib
```

## 鏁版嵁璇存槑

椤圭洰褰撳墠浣跨敤宸茬粡澶勭悊濂界殑 IoT 鎭舵剰娴侀噺鐗瑰緛鏁版嵁锛屽苟宸叉瀯寤鸿仈閭﹀涔犲鎴风鍒掑垎鐩綍銆? 
甯哥敤鏁版嵁鐩綍鍖呮嫭锛?
- `dataset/IoT`
- `dataset/IoT_20k_c20`
- `dataset/IoT_20k_c20_noniid`
- `dataset/IoT_k40_c20_noniid`

杩愯瀹為獙鍓嶏紝闇€瑕佺‘璁?`main.py` 璇诲彇鐨勬暟鎹洰褰曚笌褰撳墠鍑嗗濂界殑鏁版嵁闆嗕竴鑷淬€?
## 涓诲叆鍙?
缁熶竴涓诲叆鍙ｄ负锛?
- [src/main.py](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/src/main.py:1)

寤鸿濮嬬粓鍦?`src` 鐩綍涓嬫墽琛屽懡浠わ紝杩欐牱鐩稿璺緞鏈€绋冲畾銆?
## 蹇€熷紑濮?
### 1. 杩涘叆涓讳唬鐮佺洰褰?
```bash
cd src
```

### 2. 杩愯鍚屾瀯瀹為獙

褰撳墠浠撳簱涓殑涓変釜姝ｅ紡鑴氭湰涓哄悓鏋勫疄楠屽叆鍙ｏ細

- [run_local.sh](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/src/run_local.sh:1)
- [run_fedavg.sh](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/src/run_fedavg.sh:1)
- [run_fedproto.sh](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/src/run_fedproto.sh:1)

杩愯绀轰緥锛?
```bash
bash run_local.sh
bash run_fedavg.sh
bash run_fedproto.sh
```

### 3. 杩愯寮傛瀯瀹為獙

寮傛瀯 `MLP + CNN1D` 鐨?`FedProto` 鍏ュ彛鑴氭湰锛?
- [run_fedproto_mix_mlp_cnn1d.sh](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/src/run_fedproto_mix_mlp_cnn1d.sh:1)

杩愯鏂瑰紡锛?
```bash
bash run_fedproto_mix_mlp_cnn1d.sh
```

## 鎵嬪姩杩愯鍛戒护绀轰緥

### 鍚屾瀯 MLP + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MLP --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 鍚屾瀯 CNN1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 鍚屾瀯 Transformer + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_Transformer1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

### 寮傛瀯 MLP + CNN1D + FedProto

```bash
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

## PowerShell 杩愯鏂瑰紡

```powershell
cd src
python -u main.py -t 1 -lr 0.01 -jr 1 -lbs 10 -ls 1 -gr 1000 -eg 1 -nw 4 -nc 20 -nb 15 -dataset IoT -model_family IoT_MIX_MLP_CNN1D --input_dim 77 -fd 512 -did 0 -algo FedProto -lam 1.0 --proto_eval_mode classifier -se 100 -mart 100 -ab True --early_stop_patience 100
```

## 甯哥敤鍙傛暟璇存槑

- `-model_family`
  閫夋嫨妯″瀷瀹舵棌锛屼緥濡?`IoT_MLP`銆乣IoT_CNN1D`銆乣IoT_Transformer1D`
- `-algo`
  閫夋嫨璁粌绠楁硶锛屼緥濡?`Local`銆乣FedAvg`銆乣FedProto`
- `-fd`
  鐗瑰緛琛ㄧず缁村害锛屼篃浣滀负鍘熷瀷缁村害
- `-gr`
  鍏ㄥ眬杞暟
- `-ls`
  鏈湴璁粌杞暟
- `-nc`
  瀹㈡埛绔暟閲?- `--early_stop_patience`
  鏃╁仠瀹瑰繊杞暟
- `--skip_figures`
  璺宠繃鍥剧墖鐢熸垚锛屼粎鍦ㄩ渶瑕佽妭鐪佹椂闂存椂鎵嬪姩寮€鍚?
## 缁撴灉鐩綍璇存槑

褰撳墠缁撴灉鐩綍宸叉寜鈥滄ā鍨嬬被鍒?/ 绠楁硶 / 鏂囦欢绫诲瀷鈥濈粺涓€鏁寸悊銆?
### 鏍囧噯鐩綍缁撴瀯

- `results/MLP妯″瀷/Local/`
- `results/MLP妯″瀷/FedAvg/`
- `results/MLP妯″瀷/FedProto/`
- `results/CNN1D妯″瀷/Local/`
- `results/CNN1D妯″瀷/FedAvg/`
- `results/CNN1D妯″瀷/FedProto/`
- `results/transformer妯″瀷/Local/`
- `results/transformer妯″瀷/FedAvg/`
- `results/transformer妯″瀷/FedProto/`
- `results/寮傛瀯妯″瀷/FedProto/`

姣忎釜绠楁硶鐩綍涓嬭繘涓€姝ュ垎涓猴細

- `logs/`
  淇濆瓨 `.out` 鏃ュ織鏂囦欢
- `metrics/`
  淇濆瓨 `.h5` 鎸囨爣缁撴灉鏂囦欢
- `figures/`
  淇濆瓨 t-SNE 鍥俱€佸師鍨嬪垎甯冨浘绛夊浘鐗?
### 缁撴灉鏂囦欢鍛藉悕

鏂扮殑鏍囧噯 `.h5` 鏂囦欢鍛藉悕鏍煎紡涓猴細

```text
{dataset}_{algorithm}_{model_family}_{goal}_{run_idx}.h5
```

渚嬪锛?
```text
IoT_FedAvg_IoT_MLP_test_0.h5
IoT_FedProto_IoT_MLP_smoke_proto_0.h5
```

## 鎬荤粨鏋滆〃浣跨敤鍛戒护

璁粌瀹屾垚鍚庯紝鍙互杩愯缁熶竴姹囨€昏剼鏈壂鎻?`results/**/*.h5`锛岃嚜鍔ㄧ敓鎴愭€荤粨鏋滆〃銆?
鍦ㄩ」鐩牴鐩綍杩愯锛?
```bash
python src/summarize_results.py
```

鐢熸垚鏂囦欢锛?
- [results/summary/experiment_summary.csv](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/results/summary/experiment_summary.csv)
- [results/summary/experiment_summary.md](/c:/Users/鏈变笘璞?Desktop/姣曚笟璁捐/IoT_FedProto_thesis/results/summary/experiment_summary.md)

璇ヨ剼鏈細锛?
- 鍏煎鎵弿鏃х粨鏋滅洰褰曞拰鏂版爣鍑嗙洰褰?- 鎻愬彇 `Acc`銆乣AUC Macro`銆乣AUC Micro`銆乣Precision`銆乣Recall`銆乣F1`銆乣FNR`銆乣FPR`
- 姹囨€绘帹鐞嗗欢杩熴€侀€氫俊閲忋€佸弬鏁伴噺銆佹ā鍨嬪ぇ灏忓拰 FLOPs
- 杈撳嚭閫傚悎鍚庣画璁烘枃鏁寸悊鐨勭粺涓€缁撴灉琛?
## 缁撴灉杈撳嚭璇存槑

璁粌缁撴潫鍚庯紝绯荤粺浼氳緭鍑猴細

- `.out` 鏃ュ織鏂囦欢
- `.h5` 缁撴灉鏂囦欢
- `.png` 鍥剧墖鏂囦欢
- 姹囨€诲悗鐨?`.csv` 鍜?`.md` 鎬荤粨鏋滆〃

## 杞婚噺鍖栦笌鏁堢巼鍒嗘瀽鑴氭湰

### 妯″瀷鍙傛暟閲忕粺璁?
```bash
python scripts/report_iot_model_params.py
```

### 鑱旈偊鏁堢巼缁熻

```bash
python scripts/report_iot_efficiency.py --model-family IoT_CNN1D
```

寮傛瀯缁熻绀轰緥锛?
```bash
python scripts/report_iot_efficiency.py --model-family IoT_MIX_MLP_CNN1D
```

## 褰撳墠绯荤粺寤鸿瀹為獙缁撴瀯

### 涓诲疄楠?
- 鍚屾瀯 `MLP`锛歚Local / FedAvg / FedProto`
- 鍚屾瀯 `CNN1D`锛歚Local / FedAvg / FedProto`
- 鍚屾瀯 `Transformer`锛歚Local / FedAvg / FedProto`

### 鎵╁睍瀹為獙

- 寮傛瀯 `MLP + CNN1D`锛歚FedProto`

## 娉ㄦ剰浜嬮」

- 寤鸿濮嬬粓鍦?`src` 鐩綍杩愯涓荤▼搴忥紝閬垮厤鐩稿璺緞闂
- `FedAvg` 褰撳墠浠呯敤浜庡悓鏋勫疄楠?- 姝ｅ紡鑴氭湰榛樿浼氱敓鎴愬浘鐗囷紱鑻ュ彧鎯冲揩閫熼獙璇侊紝鍙墜鍔ㄥ姞 `--skip_figures`
- 濡傛灉 GPU 涓嶅彲鐢紝绋嬪簭浼氳嚜鍔ㄥ垏鎹㈠埌 CPU

## 褰撳墠绯荤粺瀹氫綅
璇ラ」鐩綋鍓嶅凡缁忓叿澶囷細

- 杞婚噺鍖栨湰鍦版娴嬫ā鍨?- 鑱旈偊鍘熷瀷鑱氬悎璁粌妗嗘灦
- 鍚屾瀯涓庡紓鏋勫鎴风瀹為獙鍏ュ彛
- 澶氭寚鏍囩粨鏋滆瘎浼颁笌鏁堢巼鍒嗘瀽鑳藉姏
- 缁熶竴缁撴灉鐩綍涓庢€荤粨鏋滆〃鐢熸垚鑳藉姏

鍥犳锛屽畠宸茬粡鍏峰鈥滆交閲忓寲鑱旈偊瀛︿範鎭舵剰娴侀噺妫€娴嬪師鍨嬬郴缁熲€濈殑鏍稿績缁撴瀯锛屽悗缁伐浣滀富瑕佹槸缁х画鏁寸悊缁撴灉銆佽ˉ鍏呭睍绀烘潗鏂欏苟瀹屾垚鏈€缁堣鏂囦氦浠樸€?









