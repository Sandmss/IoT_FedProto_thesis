from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import streamlit as st

try:
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
except Exception:
    get_script_run_ctx = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results"
SRC_ROOT = PROJECT_ROOT / "src"
SUMMARY_CSV = RESULTS_ROOT / "summary" / "experiment_summary.csv"

ALGORITHMS = ["Local", "FedAvg", "FedProto"]
MODEL_FAMILIES = [
    "IoT_MLP",
    "IoT_CNN1D",
    "IoT_Transformer1D",
    "IoT_MIX_MLP_CNN1D",
    "IoT_MIX_MLP_CNN_TRANS",
]
METRIC_KEYS = {
    "Accuracy": "rs_test_acc",
    "AUC Macro": "rs_test_auc_macro",
    "AUC Micro": "rs_test_auc_micro",
    "Precision": "rs_test_precision",
    "Recall": "rs_test_recall",
    "F1": "rs_test_f1",
    "FNR": "rs_test_fnr",
    "FPR": "rs_test_fpr",
    "Inference Latency (ms)": "rs_inference_latency_ms",
    "Train Loss": "rs_train_loss",
}


@dataclass
class DatasetInfo:
    name: str
    path: Path
    train_dir: Path
    test_dir: Path
    train_clients: list[str]
    test_clients: list[str]


def list_subdirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted((item for item in path.iterdir() if item.is_dir()), key=lambda item: item.name)


def list_dir_names(path: Path) -> list[str]:
    return [item.name for item in list_subdirs(path)]


@st.cache_data(show_spinner=False)
def load_dataset_catalog() -> list[DatasetInfo]:
    datasets: list[DatasetInfo] = []
    for dataset_dir in list_subdirs(DATASET_ROOT):
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"
        datasets.append(
            DatasetInfo(
                name=dataset_dir.name,
                path=dataset_dir,
                train_dir=train_dir,
                test_dir=test_dir,
                train_clients=list_dir_names(train_dir),
                test_clients=list_dir_names(test_dir),
            )
        )
    return datasets


@st.cache_data(show_spinner=False)
def load_summary_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@st.cache_data(show_spinner=False)
def discover_result_files() -> list[str]:
    files = []
    for file_path in RESULTS_ROOT.rglob("*.h5"):
        if "summary" in file_path.parts:
            continue
        files.append(file_path.relative_to(RESULTS_ROOT).as_posix())
    return sorted(files)


@st.cache_data(show_spinner=False)
def read_h5_payload(relative_path: str) -> dict[str, object]:
    file_path = RESULTS_ROOT / relative_path
    payload: dict[str, object] = {
        "path": file_path,
        "series": {},
        "best_round": None,
        "confusion_matrix": None,
    }
    with h5py.File(file_path, "r") as handle:
        series: dict[str, list[float]] = {}
        for label, key in METRIC_KEYS.items():
            if key in handle:
                series[label] = np.array(handle[key]).astype(float).tolist()
        payload["series"] = series

        acc_series = series.get("Accuracy", [])
        if acc_series:
            payload["best_round"] = int(np.argmax(np.array(acc_series)))

        confusion = handle.get("rs_confusion_matrices")
        best_round = payload["best_round"]
        if confusion is not None and best_round is not None and len(confusion) > best_round:
            payload["confusion_matrix"] = np.array(confusion[best_round]).astype(int)
    return payload


def infer_related_assets(relative_path: str) -> dict[str, list[Path]]:
    file_path = RESULTS_ROOT / relative_path
    stem = file_path.stem
    parent = file_path.parent.parent if file_path.parent.name == "metrics" else file_path.parent
    figures_dir = parent / "figures"
    logs_dir = parent / "logs"

    figure_candidates = []
    if figures_dir.exists():
        figure_candidates = sorted(figures_dir.glob(f"{stem}*.png"))

    log_candidates = []
    if logs_dir.exists():
        log_candidates = sorted(logs_dir.glob("*.out"))

    return {"figures": figure_candidates, "logs": log_candidates}


def build_training_command(config: dict[str, object]) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "-dataset",
        str(config["dataset"]),
        "-algo",
        str(config["algorithm"]),
        "-model_family",
        str(config["model_family"]),
        "-nc",
        str(config["num_clients"]),
        "-gr",
        str(config["global_rounds"]),
        "-ls",
        str(config["local_epochs"]),
        "-lr",
        str(config["local_learning_rate"]),
        "-jr",
        str(config["join_ratio"]),
        "-fd",
        str(config["feature_dim"]),
        "-dev",
        str(config["device"]),
        "-did",
        str(config["device_id"]),
        "-t",
        str(config["times"]),
        "-go",
        str(config["goal"]),
        "--input_dim",
        str(config["input_dim"]),
        "-nb",
        str(config["num_classes"]),
        "--normal_class",
        str(config["normal_class"]),
        "-lbs",
        str(config["batch_size"]),
        "-nw",
        str(config["num_workers"]),
        "--early_stop_patience",
        str(config["early_stop_patience"]),
        "-eg",
        str(config["eval_gap"]),
        "-lam",
        str(config["lamda"]),
    ]

    if config["skip_figures"]:
        command.append("--skip_figures")

    if config["model_family"] in {"IoT_Transformer1D", "IoT_MIX_MLP_CNN_TRANS"}:
        command.extend(
            [
                "--transformer_d_model",
                str(config["transformer_d_model"]),
                "--transformer_num_heads",
                str(config["transformer_num_heads"]),
                "--transformer_num_layers",
                str(config["transformer_num_layers"]),
                "--transformer_dropout",
                str(config["transformer_dropout"]),
            ]
        )

    return command


def rerun_summary() -> tuple[bool, str]:
    command = [sys.executable, "src/summarize_results.py"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    load_summary_rows.clear()
    discover_result_files.clear()
    read_h5_payload.clear()
    return completed.returncode == 0, output.strip()


def render_metric_cards(metrics: list[tuple[str, float | None, str]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics):
        display_value = "-" if value is None else f"{value:.4f}"
        column.metric(label, display_value, help=help_text)


def metric_value(series: dict[str, list[float]], label: str, reducer: str = "max") -> float | None:
    values = series.get(label, [])
    if not values:
        return None
    array = np.array(values, dtype=float)
    if reducer == "min":
        return float(array.min())
    if reducer == "mean":
        return float(array.mean())
    return float(array.max())


def render_result_detail(relative_path: str) -> None:
    payload = read_h5_payload(relative_path)
    series = payload["series"]
    best_round = payload["best_round"]
    confusion_matrix = payload["confusion_matrix"]
    assets = infer_related_assets(relative_path)

    st.markdown(f"#### 实验文件")
    st.code(relative_path, language="text")

    render_metric_cards(
        [
            ("最佳 Accuracy", metric_value(series, "Accuracy", "max"), "测试集准确率峰值"),
            ("最佳 AUC Macro", metric_value(series, "AUC Macro", "max"), "宏平均 AUC 峰值"),
            ("最佳 F1", metric_value(series, "F1", "max"), "F1 峰值"),
            ("平均时延", metric_value(series, "Inference Latency (ms)", "mean"), "平均单次推理时延"),
        ]
    )

    st.caption(f"最佳轮次: {best_round if best_round is not None else '-'}")

    chart_labels = ["Accuracy", "AUC Macro", "F1", "FNR", "FPR", "Train Loss"]
    chart_data = {label: series[label] for label in chart_labels if label in series and series[label]}
    if chart_data:
        st.line_chart(chart_data, height=320)
    else:
        st.info("当前结果文件中没有可绘制的时序指标。")

    if confusion_matrix is not None:
        st.markdown("#### 最佳轮次混淆矩阵")
        st.dataframe(confusion_matrix, width="stretch")

    if assets["figures"]:
        st.markdown("#### 关联图像")
        for figure_path in assets["figures"]:
            st.image(str(figure_path), caption=figure_path.name, width="stretch")

    if assets["logs"]:
        log_path = assets["logs"][0]
        st.markdown("#### 训练日志摘要")
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        preview = "\n".join(lines[-30:]) if lines else "未读取到日志内容。"
        st.code(preview, language="text")


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f1e8;
            --card: rgba(255, 252, 247, 0.92);
            --ink: #1d2a31;
            --muted: #5d6d75;
            --accent: #0f766e;
            --accent-2: #c2410c;
            --line: rgba(29, 42, 49, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(194,65,12,0.10), transparent 24%),
                linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(29, 42, 49, 0.05);
        }
        .hero-card {
            padding: 1.2rem 1.3rem;
            background: linear-gradient(135deg, rgba(255,252,247,0.95), rgba(255,246,235,0.92));
            border: 1px solid var(--line);
            border-radius: 22px;
            margin-bottom: 1rem;
            box-shadow: 0 14px 38px rgba(29, 42, 49, 0.07);
        }
        .hero-card h1 {
            margin: 0;
            font-size: 2rem;
        }
        .hero-card p {
            color: var(--muted);
            margin: 0.5rem 0 0 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def running_with_streamlit() -> bool:
    if get_script_run_ctx is None:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__" and not running_with_streamlit():
    print("This app must be started with Streamlit.")
    print("Use one of the following commands:")
    print("  streamlit run app.py")
    print("  python -m streamlit run app.py")
    raise SystemExit(1)


st.set_page_config(
    page_title="IoT FedProto Visual Workspace",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_styles()

datasets = load_dataset_catalog()
dataset_names = [dataset.name for dataset in datasets]
default_dataset = dataset_names[0] if dataset_names else None

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = default_dataset

st.markdown(
    """
    <div class="hero-card">
        <h1>IoT FedProto 可视化实验工作台</h1>
        <p>围绕数据选择、联邦训练、结果解析与历史对比，组织现有 Python 实验链路，形成可本地运行的论文演示界面。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("实验上下文")
    if dataset_names:
        selected_dataset_name = st.selectbox("数据集", dataset_names, index=dataset_names.index(st.session_state.selected_dataset))
        st.session_state.selected_dataset = selected_dataset_name
        dataset_info = next(dataset for dataset in datasets if dataset.name == selected_dataset_name)
    else:
        dataset_info = None
        st.warning("未发现可用数据集目录。")

    st.markdown("### 路径状态")
    st.write(f"`dataset/`: {'存在' if DATASET_ROOT.exists() else '缺失'}")
    st.write(f"`results/`: {'存在' if RESULTS_ROOT.exists() else '缺失'}")
    st.write(f"`src/`: {'存在' if SRC_ROOT.exists() else '缺失'}")

    if dataset_info:
        st.markdown("### 当前数据摘要")
        st.write(f"训练客户端数: `{len(dataset_info.train_clients)}`")
        st.write(f"测试客户端数: `{len(dataset_info.test_clients)}`")
        st.write(f"训练目录: `{dataset_info.train_dir}`")
        st.write(f"测试目录: `{dataset_info.test_dir}`")

    if st.button("刷新结果汇总", width="stretch"):
        ok, output = rerun_summary()
        if ok:
            st.success("结果汇总已刷新。")
        else:
            st.error("结果汇总刷新失败。")
        if output:
            st.code(output, language="text")

tab_dataset, tab_train, tab_result, tab_history = st.tabs(
    ["数据与客户端", "联邦训练", "结果解析", "历史结果"]
)

with tab_dataset:
    if not dataset_info:
        st.info("请先准备 `dataset/` 目录。")
    else:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.subheader("数据集结构")
            st.write(f"数据集根目录: `{dataset_info.path}`")
            client_selection = st.multiselect(
                "参与训练的客户端",
                options=dataset_info.train_clients,
                default=dataset_info.train_clients,
                help="当前界面以本地目录中的客户端子文件夹作为实验参与方。",
            )
            st.session_state.selected_clients = client_selection
            render_metric_cards(
                [
                    ("已选客户端", float(len(client_selection)) if client_selection else None, "当前训练任务使用的客户端数"),
                    ("训练集目录", float(len(dataset_info.train_clients)), "可用训练客户端总数"),
                    ("测试集目录", float(len(dataset_info.test_clients)), "可用测试客户端总数"),
                ]
            )

        with right:
            st.subheader("目录浏览")
            client_col1, client_col2 = st.columns(2)
            client_col1.markdown("**Train Clients**")
            client_col1.dataframe(dataset_info.train_clients, width="stretch")
            client_col2.markdown("**Test Clients**")
            client_col2.dataframe(dataset_info.test_clients, width="stretch")

        st.subheader("实验说明")
        st.info(
            "当前版本采用仓库内现有 Python 实验代码作为执行核心，界面负责组织数据选择、参数配置、训练触发和结果展示，符合开发计划中的“最小侵入式集成”思路。"
        )

with tab_train:
    if not dataset_info:
        st.info("没有可用数据集，暂时无法配置训练任务。")
    else:
        selected_clients = st.session_state.get("selected_clients", dataset_info.train_clients)
        st.subheader("训练配置")
        with st.form("train_form"):
            col1, col2, col3 = st.columns(3)
            algorithm = col1.selectbox("算法", ALGORITHMS, index=2)
            model_family = col2.selectbox("模型", MODEL_FAMILIES, index=0)
            device = col3.selectbox("设备", ["cpu", "cuda"], index=0)

            col4, col5, col6 = st.columns(3)
            num_clients = col4.number_input("客户端数量", min_value=1, value=max(1, len(selected_clients)))
            global_rounds = col5.number_input("全局轮数", min_value=1, value=100)
            local_epochs = col6.number_input("本地 Epoch", min_value=1, value=1)

            col7, col8, col9 = st.columns(3)
            local_learning_rate = col7.number_input("学习率", min_value=0.0001, value=0.0050, format="%.4f")
            join_ratio = col8.number_input("Join Ratio", min_value=0.1, max_value=1.0, value=1.0, format="%.2f")
            feature_dim = col9.number_input("Feature Dim", min_value=8, value=64)

            col10, col11, col12 = st.columns(3)
            batch_size = col10.number_input("Batch Size", min_value=1, value=10)
            num_workers = col11.number_input("DataLoader Workers", min_value=0, value=0)
            lamda = col12.number_input("Lambda", min_value=0.0, value=1.0, format="%.2f")

            col13, col14, col15 = st.columns(3)
            input_dim = col13.number_input("Input Dim", min_value=1, value=77)
            num_classes = col14.number_input("类别数", min_value=2, value=15)
            normal_class = col15.number_input("Normal Class", min_value=0, value=0)

            col16, col17, col18 = st.columns(3)
            early_stop_patience = col16.number_input("早停轮次", min_value=1, value=100)
            eval_gap = col17.number_input("Eval Gap", min_value=1, value=1)
            times = col18.number_input("重复次数", min_value=1, value=1)

            col19, col20, col21 = st.columns(3)
            transformer_d_model = col19.number_input("Transformer d_model", min_value=8, value=64)
            transformer_num_heads = col20.number_input("Transformer heads", min_value=1, value=4)
            transformer_num_layers = col21.number_input("Transformer layers", min_value=1, value=2)

            col22, col23, col24 = st.columns(3)
            transformer_dropout = col22.number_input("Transformer dropout", min_value=0.0, max_value=0.9, value=0.2, format="%.2f")
            device_id = col23.text_input("CUDA Device ID", value="0")
            goal = col24.text_input("Goal Tag", value="test")

            skip_figures = st.checkbox("跳过图像生成", value=False)
            submitted = st.form_submit_button("生成并执行训练")

        config = {
            "dataset": dataset_info.name,
            "algorithm": algorithm,
            "model_family": model_family,
            "num_clients": int(num_clients),
            "global_rounds": int(global_rounds),
            "local_epochs": int(local_epochs),
            "local_learning_rate": float(local_learning_rate),
            "join_ratio": float(join_ratio),
            "feature_dim": int(feature_dim),
            "device": device,
            "device_id": device_id,
            "times": int(times),
            "goal": goal,
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "normal_class": int(normal_class),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "early_stop_patience": int(early_stop_patience),
            "eval_gap": int(eval_gap),
            "lamda": float(lamda),
            "skip_figures": skip_figures,
            "transformer_d_model": int(transformer_d_model),
            "transformer_num_heads": int(transformer_num_heads),
            "transformer_num_layers": int(transformer_num_layers),
            "transformer_dropout": float(transformer_dropout),
        }
        command = build_training_command(config)

        st.markdown("#### 训练命令预览")
        st.code(subprocess.list2cmdline(command), language="bash")
        st.caption(f"当前选中的客户端数: {len(selected_clients)}；实际执行会把 `-nc` 设置为上方表单中的值。")

        if submitted:
            st.markdown("#### 训练日志")
            log_placeholder = st.empty()
            status_placeholder = st.empty()
            logs: list[str] = []

            process = subprocess.Popen(
                command,
                cwd=SRC_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            status_placeholder.info("训练任务已启动，正在读取实时日志。")
            assert process.stdout is not None
            for line in process.stdout:
                logs.append(line.rstrip())
                log_placeholder.code("\n".join(logs[-120:]), language="text")

            return_code = process.wait()
            if return_code == 0:
                status_placeholder.success("训练完成，正在刷新结果索引。")
                rerun_summary()
                st.success("训练与结果汇总已完成刷新。")
            else:
                status_placeholder.error(f"训练失败，退出码: {return_code}")

with tab_result:
    result_files = discover_result_files()
    if not result_files:
        st.info("暂未发现可解析的 `.h5` 结果文件。")
    else:
        st.subheader("单次实验结果解析")
        chosen_result = st.selectbox("选择结果文件", result_files)
        render_result_detail(chosen_result)

with tab_history:
    rows = load_summary_rows()
    if not rows:
        st.info("暂无历史汇总表，请先在侧边栏点击“刷新结果汇总”。")
    else:
        st.subheader("历史实验对比")
        algorithms = sorted({row["Algorithm"] for row in rows if row.get("Algorithm")})
        models = sorted({row["Model"] for row in rows if row.get("Model")})
        col1, col2 = st.columns(2)
        algorithm_filter = col1.multiselect("筛选算法", algorithms, default=algorithms)
        model_filter = col2.multiselect("筛选模型", models, default=models)

        filtered_rows = [
            row for row in rows
            if row.get("Algorithm") in algorithm_filter and row.get("Model") in model_filter
        ]
        filtered_rows.sort(key=lambda row: float(row.get("Acc", 0.0)), reverse=True)

        if filtered_rows:
            top = filtered_rows[0]
            render_metric_cards(
                [
                    ("最佳 Accuracy", float(top["Acc"]), "当前筛选条件下的最高准确率"),
                    ("最佳模型", None, top["Model"]),
                    ("最佳算法", None, top["Algorithm"]),
                    ("记录条数", float(len(filtered_rows)), "当前筛选后的实验数量"),
                ]
            )
            st.dataframe(filtered_rows, width="stretch")

            preview_options = [row["ResultFile"] for row in filtered_rows if row.get("ResultFile")]
            if preview_options:
                st.markdown("#### 结果详情联动预览")
                preview_target = st.selectbox("选择历史结果进行展开", preview_options, key="history_preview")
                render_result_detail(preview_target)
        else:
            st.warning("筛选后没有可展示的实验记录。")
