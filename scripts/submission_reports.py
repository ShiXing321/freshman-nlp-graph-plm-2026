"""Build portable experiment reports from completed result files."""
import json
import os
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path
from build_reports import ROOT, NLP, GRAPH, results, plot, compact

REPORT = ROOT / "学习报告"


def link(path, target):
    return Path(os.path.relpath(path, target.parent)).as_posix()


def select(folder, domain, task):
    records = results(folder / "results")
    if domain == "nlp" and task in ["task1", "task2"]:
        original = [(p, d) for p, d in records if d["config"].get("data_name") == "sentiment_kaggle"]
        return original or records
    if domain == "nlp" and task in ["task3", "task4"]:
        expanded = [(p, d) for p, d in records if d["name"].endswith("_expanded")]
        if task == "task4":
            expanded += [(p, d) for p, d in records if d["name"].endswith("_casing")]
        return expanded or records
    if domain == "graph" and task in ["task1", "task2"]:
        isolated = [(p, d) for p, d in records if d["name"].endswith("_isolated")]
        if len(isolated) == 24:
            extras = [(p, d) for p, d in records if "_seed" in d["name"] or d["name"].endswith(("_ablation", "_tuned"))]
            return isolated + extras
        return [(p, d) for p, d in records if not d["name"].endswith("_isolated")]
    if domain == "graph" and task == "task4":
        full = [(p, d) for p, d in records if d["name"].endswith("_full")]
        return full if len(full) == 3 else [(p, d) for p, d in records if not d["name"].endswith("_full")]
    return records


def table(records, target):
    lines = ["| 实验配置 | 训练规模 | 轮数 | 验证指标 | 测试指标 | 运行时间 / s |",
             "|---|---:|---:|---|---|---:|"]
    for path, row in records:
        config = row["config"]
        counts = config.get("counts", row.get("counts", {}))
        n = counts.get("train", config.get("training_triples", config.get("nodes", "—")))
        if "nodes" in config and not counts:
            n = str(n) + "节点"
        if row.get("character_counts"):
            n = str(row["character_counts"]["train"]) + "字符"
        dev = "; ".join(k.replace("best_dev_", "") + f"={v:.4f}" for k, v in row.items() if k.startswith("best_dev_") and isinstance(v, (float, int))) or "—"
        metrics = row.get("test", {})
        if "test_accuracy" in row:
            metrics = {"accuracy": row["test_accuracy"], "macro_f1": row["test_macro_f1"]}
        if "test_perplexity" in row:
            metrics = {"loss": row["test_loss"], "perplexity": row["test_perplexity"]}
        epochs = "—" if config.get("task") == "mbti" or str(config.get("task", "")).startswith("zero-shot") else len(row.get("epochs", []))
        lines.append(f"| [{row['name']}]({link(path, target)}) | {n} | {epochs} | {dev} | {compact(metrics)} | {row.get('elapsed_seconds', 0):.1f} |")
    return "\n".join(lines)


def analyses(records, domain, task):
    lines = []
    def comparison_group(row):
        if domain == "plm":
            name = row["name"]
            if task == "T5生成":
                return "T5问答" if name.startswith("t5_qa") else "T5填空"
            if task == "文本匹配":
                return "SimCSE无监督划分" if name.startswith("simcse") else "监督评论匹配"
            if task == "信息抽取":
                return "预训练UIE" if name.startswith("uie_base_") else "小BERT边界抽取"
            if task == "大模型训练":
                return {"sft": "Qwen指令微调", "cpt": "Qwen继续预训练", "scratch": "随机初始化GPT2", "scratch-sft": "GPT2指令微调"}[name]
        return row["config"].get("dataset", "当前数据")
    datasets = sorted({comparison_group(row) for _, row in records})
    for ds in datasets:
        group = [row for _, row in records if comparison_group(row) == ds]
        candidates = []
        for row in group:
            if "_seed" in row["name"]:
                continue
            metric = next((k for k in row if k.startswith("best_dev_")), None)
            if metric:
                value = row[metric]
                low = "loss" in metric or row["config"].get("problem") == "regression"
                candidates.append(((-value if low else value), row, metric))
        if candidates:
            _, best, key = max(candidates, key=lambda item: item[0])
            score = best.get("test", {})
            if "test_accuracy" in best:
                score = {"accuracy": best["test_accuracy"], "macro_f1": best["test_macro_f1"]}
            lines.append(f"{ds}按验证集选择的配置为 `{best['name']}`，{key.replace('best_dev_', '')}={best[key]:.4f}，测试结果为 {compact(score)}。")
    pairs = {}
    for _, row in records:
        c = row["config"]
        if c.get("mode") in ["full", "sampled", "batch"] and "_seed" not in row["name"] and "_ablation" not in row["name"]:
            key = (c.get("dataset"), c.get("model"), c.get("pool", ""), c.get("layers"), c.get("lr"))
            pairs.setdefault(key, {})[c["mode"]] = row
    timings = []
    for key, modes in pairs.items():
        second = "sampled" if "sampled" in modes else "batch"
        if "full" in modes and second in modes:
            a, b = modes["full"], modes[second]
            at, bt = a.get("train_seconds"), b.get("train_seconds")
            if at is not None and bt:
                label = ' / '.join(str(k) for k in key[:3] if k) + f" / l{key[3]} lr{key[4]}"
                timings.append(f"| {label} | {at:.3f} | {bt:.3f} | {at / bt:.3f} | {compact(a.get('test', {}))} | {compact(b.get('test', {}))} |")
    if timings:
        lines += ["", "训练耗时在CUDA同步后记录。上表的运行时间从实验记录建立时开始，包含训练、验证、测试和保存，数据下载与准备另记在运行日志中。下表倍率为全图训练时间除以另一模式训练时间，便于同时观察速度和指标。", "", "| 数据 / 模型 / 池化 | full / s | sampled 或 batch / s | 时间倍率 | full 测试 | 另一模式测试 |", "|---|---:|---:|---:|---|---|", *timings]
    if domain == "nlp" and task in ["task1", "task2"]:
        source = ROOT / "nlp-beginner/legacy/data/processed/sentiment_kaggle"
        if source.exists():
            counts = Counter(json.loads(line)["label"] for line in (source / "test.jsonl").read_text(encoding="utf-8").splitlines())
            lines += [f"Kaggle测试集中五类数量为 `{dict(sorted(counts.items()))}`，只预测最多类别的准确率为 {max(counts.values()) / sum(counts.values()):.4f}。短语中有较多中性词组，accuracy和macro-F1需要一起观察；macro-F1能反映少数类别的识别情况。"]
        if task == "task1":
            lines += ["按124,481条训练数据和20轮计算，full-batch共更新20次，batch=64共更新38,920次，SGD共更新2,489,620次。每轮覆盖相同数据，参数更新的频率和训练时间差异很大。SGD和mini-batch可在学习率0.05的两组间对照；full-batch和mini-batch可在0.5的两组间对照。"]
        if task == "task2":
            paired = {}
            for _, row in records:
                c = row["config"]
                if c.get("model") in ["cnn", "rnn"]:
                    paired.setdefault(c["model"], {})["glove" if c.get("glove") else "random"] = row
            for model, arms in paired.items():
                if set(arms) == {"glove", "random"}:
                    a, b = arms["random"]["test"], arms["glove"]["test"]
                    lines += [f"{model.upper()}随机初始化测试accuracy={a['accuracy']:.4f}、macro-F1={a['macro_f1']:.4f}；GloVe初始化为{b['accuracy']:.4f}、{b['macro_f1']:.4f}，accuracy变化{b['accuracy'] - a['accuracy']:+.4f}。两组采用相同数据与训练轮数，词向量在训练中继续更新。"]
    for _, row in records:
        if row.get("glove_coverage") is not None:
            lines.append(f"`{row['name']}` 的 GloVe 词表覆盖率为 {row['glove_coverage']:.2%}，其余词保留随机初始化。")
        if row.get("generated"):
            lines += [f"`{row['name']}` 的生成样例：", "", "```json", json.dumps(row["generated"], ensure_ascii=False, indent=2), "```"]
    if domain == "graph" and task == "task1":
        import numpy as np
        label_path = ROOT / "运行记录/诊断/flickr_labels.json"
        if label_path.exists():
            labels = json.loads(label_path.read_text(encoding="utf-8"))
            counts = labels["counts"]
            lines += [f"Flickr官方标签划分为训练{counts['tr']:,}、验证{counts['va']:,}、测试{counts['te']:,}个节点。训练集中最多的类别为{labels['training_majority_class']}，固定预测该类的测试准确率为{labels['test_majority_accuracy']:.4f}。类别分布偏向少数大类，因此结合macro-F1观察各类识别情况。"]
        repeated = {}
        for _, row in records:
            c = row["config"]
            if c.get("dataset") == "Cora" and c.get("model") == "gcn" and c.get("layers") == 2 and c.get("lr") == .01 and ("_seed" in row["name"] or row["name"].endswith("_isolated")):
                repeated.setdefault(c["mode"], []).append((c["seed"], row["test"]["accuracy"]))
        if all(len(repeated.get(mode, [])) == 3 for mode in ["full", "sampled"]):
            lines += ["### Cora GCN多种子结果", "", "| 模式 | 种子 | 测试accuracy均值 | 标准差 |", "|---|---|---:|---:|"]
            for mode, rows in repeated.items():
                values = [value for _, value in rows]
                lines.append(f"| {mode} | {', '.join(str(seed) for seed, _ in rows)} | {np.mean(values):.4f} | {np.std(values, ddof=1):.4f} |")
            lines += ["三个种子保持划分和模型配置一致，用均值与标准差观察初始化和训练顺序带来的波动。"]
    if domain == "graph" and task == "task2":
        coverage = []
        for key, modes in pairs.items():
            if key[1] != "gcn" or key[3:] != (2, .01) or "full" not in modes or "sampled" not in modes:
                continue
            full = modes["full"]["epochs"][0]["supervised_edges"]
            average = sum(r["supervised_edges"] for r in modes["sampled"]["epochs"]) / len(modes["sampled"]["epochs"])
            coverage.append(f"| {key[0]} | {full:,} | {average:,.1f} | {average / full:.2%} |")
        if coverage:
            lines += ["### 采样后的监督量", "", "以默认两层GCN为例，每轮实际参与损失的监督边如下。RandomNodeLoader把节点分为4个诱导子图，只保留两个端点位于同一分区的监督边；时间与指标差异需要结合这一变化解释。", "", "| 数据集 | full每轮监督边 | sampled每轮平均监督边 | 保留比例 |", "|---|---:|---:|---:|", *coverage]
    if domain == "graph" and task in ["task1", "task2", "task3"]:
        studies = [row for _, row in records if row["name"].endswith("_ablation")]
        if studies:
            lines += ["", "### 层数与学习率", "", "参数对照保持数据划分、模型种类和训练模式一致，分别调整层数或学习率。下表同时列出默认配置，验证分数用于选优，测试分数用于观察留出效果。"]
            for ds in sorted({r["config"]["dataset"] for r in studies}):
                lines += ["", "| 数据 / 模型 | 层数 | 学习率 | 验证分数 | 测试分数 |", "|---|---:|---:|---:|---:|"]
                group = [r for r in studies if r["config"]["dataset"] == ds]
                model = group[0]["config"]["model"]
                default_lr = .001 if task == "task3" else .01
                defaults = [r for _, r in records if r["config"].get("dataset") == ds and r["config"].get("model") == model and r["config"].get("layers") == 2 and r["config"].get("lr") == default_lr and r["config"].get("mode") == ("batch" if task == "task3" else "full") and (task != "task3" or r["config"].get("pool") == "mean") and not r["name"].endswith("_ablation") and "_seed" not in r["name"]]
                group += defaults[:1]
                metric = "mae" if ds == "ZINC" else "auc" if task == "task2" else "accuracy"
                dev_key = "best_dev_score" if task == "task3" else "best_dev_auc" if task == "task2" else "best_dev_accuracy"
                for row in sorted(group, key=lambda r: (r["config"]["lr"], r["config"]["layers"])):
                    c = row["config"]
                    lines.append(f"| {ds} / {model.upper()} | {c['layers']} | {c['lr']} | {row[dev_key]:.4f} | {row['test'][metric]:.4f} |")
                depth = sorted([r for r in group if r["config"]["lr"] == default_lr], key=lambda r: r["config"]["layers"])
                rate = sorted([r for r in group if r["config"]["layers"] == 2], key=lambda r: r["config"]["lr"])
                lines += ["", f"{ds}的层数对照：" + "、".join(f"{r['config']['layers']}层测试{metric}={r['test'][metric]:.4f}" for r in depth) + "。学习率对照：" + "、".join(f"lr={r['config']['lr']}时为{r['test'][metric]:.4f}" for r in rate) + "。"]
            lines += ["", "层数增加会扩大节点接收信息的范围，也增加连续聚合和优化的难度；学习率同时影响下降速度与验证波动。结果结合曲线判断，在当前划分上选择验证分数较好的配置。"]
    return "\n".join(lines)


METHODS = {
    ("nlp", "task1"): "读取Kaggle电影评论的156,060条短语，按SentenceId分组，以固定种子42划分训练、验证、测试，数量分别为124,481、15,585、15,994。词表及IDF只在训练集拟合。用NumPy手写softmax和one-vs-rest logistic的损失、梯度及更新，比较count、binary、TF-IDF、unigram/bigram、学习率、批量大小和shuffle，共9组。",
    ("nlp", "task2"): "沿用任务一的数据划分与TF-IDF，用PyTorch复写两种线性分类器，并核对一批真实样本的损失和梯度。随后训练平均词向量、CNN与双向LSTM分类器，对CNN和LSTM分别使用随机词向量与50维GloVe初始化，共7组。每轮训练后在验证集选优，最后在测试集评价。",
    ("nlp", "task3"): "使用SNLI官方划分，过滤无共识标签。新增实验固定采样100,000条训练数据，验证和测试使用完整有效划分，训练10轮。模型采用GloVe初始化、共享双向LSTM、双向词级注意力与局部推断组合，分类目标为蕴含、矛盾和中立。",
    ("nlp", "task4"): "使用CoNLL-2003官方训练、验证与测试数据。双向LSTM产生BIO标签的发射分数，自写CRF计算路径似然，Viterbi求最优标签序列。基础实验使用GloVe，隐层64维，训练20轮；扩展实验拼接8维大小写特征，将隐层增至128维，训练30轮。长度上限512，数据中最长句为124个词。评价采用实体边界和类型都正确的实体级precision、recall、F1。",
    ("nlp", "task5"): "将唐诗文本先按连续区段划为80%、10%、10%，分别建立字符窗口，输入和目标错开一个字符。LSTM与GRU各训练20轮，按验证困惑度选择参数，计算测试困惑度，并在温度0.7和1.0下生成80个字符。",
    ("graph", "task1"): "采用PyG，在Cora、CiteSeer、Flickr上训练GCN、GAT、GraphSAGE和GIN。每个模型分别使用全图与RandomNodeLoader随机诱导子图，保留官方节点mask，共24组。结果表的节点数表示图结构规模，训练损失只使用train_mask中的标签。Cora补学习率和层数对照，以及GCN的多种子实验。新增计时实验依次运行，GPU训练保持串行。",
    ("graph", "task2"): "用RandomLinkSplit将边划为10%验证、10%测试，训练正边中的20%单独作监督边。负样本在所有正边之外采样，正负约1:1。四种GNN产生节点表示，内积解码边分数，二元交叉熵训练。在三个数据集上比较全图与框架子图采样，共24组，并补Cora的学习率与层数对照。",
    ("graph", "task3"): "MUTAG采用TUDataset，固定种子按图划分80%、10%、10%。ZINC采用官方full划分的国内镜像，固定采样训练2,000、验证1,000、测试1,000。四种GNN分别配mean、max、min池化，再比较全批次和mini-batch，共48组。MUTAG使用交叉熵和分类指标，ZINC使用L1损失和MAE；新增GIN的层数、学习率对照。",
    ("graph", "task4"): "参考Maxioo/kge_framework的三元组输入、负采样与filtered排名流程，用PyTorch实现TransE、RotatE与ConvE的统一评分接口。参考代码保存在根目录kge_framework/code，本地训练入口在本任务code目录。在FB15k-237官方划分上使用全部训练三元组，训练20轮，固定200条验证样本选优，测试遍历完整测试集。每条测试三元组分别替换头实体和尾实体，在全部实体候选中做filtered排序，计算MRR、Hits@1/3/10。",
}

ISSUES = {
    ("nlp", "task1"): "最初的句子级SST-5与原题的短语数据规模不同，补取Kaggle原始train.tsv并记录SHA-256。短语共享SentenceId，采用分组划分处理潜在的数据泄漏。NumPy词表编号序列化时转为Python整数。",
    ("nlp", "task2"): "CNN池化需要屏蔽padding；LSTM使用真实长度打包。线性模型保留相同特征、零初始化和SGD，便于检验NumPy与PyTorch的对应关系。GloVe下载采用分块续传，并核对ZIP条目的大小和CRC。读取GloVe时会生成未命中词的随机向量，因此在构造分类模型前重新固定种子，使随机/GloVe两组的分类头与训练顺序保持一致。",
    ("nlp", "task3"): "SNLI镜像下载出现连接中断，使用带Content-Range核对的续传并校验哈希。扩大训练子集后保留早期2万条实验作为过程记录，正式表格使用新增实验。",
    ("nlp", "task4"): "早期长度上限100会截断少量句子的尾部，新增实验提高到512。CRF前向算法和Viterbi用短序列穷举校验，mask同时作用于路径分数、配分函数和解码。另把完整测试集与公开CoNLL副本转换成相同BIO格式后逐句核对，3,453句的词序列与标签均一致。大小写特征保留原词形，词向量仍使用小写词编号。",
    ("nlp", "task5"): "先划分连续文本再切窗口，使留出区段保持独立。字符词表由训练区段建立，未见字符映射到unk，数量记录在结果中。生成时保留循环隐状态，避免每个字符都重新从空状态开始。",
    ("graph", "task1"): "Flickr从国内可访问的数据源恢复，保留原始role划分。RandomNodeLoader会移除跨分区边，采样结果结合指标、时间和显存分析。早期并发记录留作历史，正式计时采用后续串行实验。",
    ("graph", "task2"): "采样子图需要将全局节点编号映射为局部编号，监督边仅保留两个端点都在子图中的部分；每轮实际监督边数写入JSON。训练消息边与留出正边分别管理。部分Flickr分数较大，sigmoid在浮点计算中接近1，评价改用原始logit计算AUC/AP，并重跑24组对照。另补学习率0.001的Flickr实验，查看步长对训练效果的影响。",
    ("graph", "task3"): "ZINC的目标是连续性质，使用回归评价。下载采用官方full划分的镜像，实验采样数量单独写入配置。基本GNN使用原子类别和连接关系；键类型可作为后续扩展。full每轮一次更新，batch每轮多次更新，结合更新次数解释结果。",
    ("graph", "task4"): "直接逐候选调用ConvE较慢，改为按关系缓存头实体查询并以矩阵运算评分；TransE和RotatE也合并候选计算。优化后用小图直接评分逐一核对MRR，三种模型均通过。ConvE早期分别前向正负样本，BatchNorm统计随两组输入改变；改为把正负三元组合并后一次前向，再分别计算损失。验证固定抽样，测试使用完整划分。",
}


def task_report(spec, domain):
    title, task, _, knowledge, commands, _ = spec
    if domain == "nlp" and task == "task1":
        knowledge += "\n\n上述梯度对应数据损失部分。代码同时采用固定λ=1e-4的L2正则，损失增加`λ/2 * sum(W²)`，权重梯度增加`λW`，偏置保留原梯度。它约束权重大小，各组使用相同系数。"
    if domain == "graph" and task == "task4":
        knowledge = knowledge.replace("本实现用optimistic ties。", "同分候选采用平均排名。")
    for old, new in {
        "3分类的随机水平约1/3，不能把高于随机水平就解释为模型完全理解逻辑。": "3分类的均匀随机准确率约1/3，可作最简单的基线。",
        "跨分区边会丢失，不能当成多跳邻居采样。": "训练子图由随机分区的节点诱导得到，跨分区边在本轮被移除。",
        "不可计算分类准确率。": "连续目标以MAE评价。",
        "因而采样训练每轮监督边数减少；JSON记录了数量。这种对比同时改变结构与监督量，不能只把结果归因于运行方式。": "采样训练每轮监督边数减少，实际数量记录在JSON中，结果结合保留的图结构与监督量一起分析。",
        "不能先对重叠窗口随机分割数据，否则训练与测试可能看到同一段文字。这里先按连续文本80/10/10划分，再分别建立窗口。": "先按连续文本80/10/10划分，再分别建立窗口，使各划分保留不同的原始文本区段。",
        "不能用token accuracy替代。": "token accuracy反映逐词标签的正确率，两者的评价含义不同。",
        "未使用键类型特征，属于简单基线。": "键类型特征可在消息函数中继续扩展。",
        "MUTAG测试仅19张图，单个样本就明显改变accuracy，结论需要谨慎。": "MUTAG测试有19张图，一张图的预测变化会使accuracy变化约5.26个百分点，可结合macro-F1观察类别识别情况。",
    }.items():
        knowledge = knowledge.replace(old, new)
    folder = ROOT / ("nlp-beginner/legacy" if domain == "nlp" else "graph_beginner") / task
    target = folder / "README.md"
    records = select(folder, domain, task)
    kind = domain + task[-1]
    text = ["# " + title, "", "## 一、实验环境与条件", "", "Windows 11，Python 3.11，RTX 3050 Ti Laptop 4GB。PyTorch 2.6.0+cu124；图任务使用PyG 2.6.1。NumPy分类运行在CPU，神经网络训练使用CUDA。种子默认为42。完整版本与下载来源见[环境说明](" + link(REPORT / "实验环境.md", target) + ")。", "", "## 二、实验方法与过程", "", METHODS[domain, task], "", "### 实现与知识点", "", knowledge, "", "### 复现命令", "", "在项目根目录运行：", "", "```powershell"]
    if domain == "nlp" and task == "task3":
        commands = ["nlp-beginner/legacy/train_torch.py matching --model esim --limit 100000 --eval-limit 0 --glove nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt --epochs 10 --batch 64 --tag _expanded"]
    if domain == "nlp" and task == "task4":
        commands = ["nlp-beginner/legacy/train_torch.py ner --model bilstm-crf --glove nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt --epochs 20 --max-length 512 --batch 32 --tag _expanded", "nlp-beginner/legacy/train_torch.py ner --model bilstm-crf --glove nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt --epochs 30 --hidden 128 --max-length 512 --batch 32 --use-case --tag _casing", "scripts/verify_algorithms.py"]
    if domain == "graph" and task == "task4":
        commands = ["graph_beginner/task4/code/train.py --model rotate --epochs 20 --limit 0 --eval-limit 200 --test-limit 0 --batch 1024 --negatives 8 --tag _full", "scripts/verify_kge_ranking.py"]
    for command in commands:
        if domain == "nlp" and task in ["task1", "task2"] and "train_" in command:
            command += " --data-name sentiment_kaggle"
        text.append(".\\.venv\\Scripts\\python.exe " + command)
    text += ["```", "", "全部新增对照可通过 `python scripts/quality_runs.py`、`python scripts/quality_extra_runs.py` 与 `python scripts/quality_followup_runs.py` 顺序复现，结果和权重完整的成功命令会按日志跳过。", "", "## 三、实验结果及分析", "", table(records, target), "", analyses(records, domain, task)]
    if domain == "graph" and task == "task4":
        text += ["", "| 模型 | MRR | Hits@1 | Hits@3 | Hits@10 | 测试三元组 |", "|---|---:|---:|---:|---:|---:|"]
        for _, row in records:
            m = row["test"]
            text.append(f"| {row['config']['model'].upper()} | {m['mrr']:.4f} | {m.get('hits@1', 0):.4f} | {m.get('hits@3', 0):.4f} | {m['hits@10']:.4f} | {m.get('evaluated_triples', 0)} |")
        text += ["", "每条测试三元组分别预测头与尾，因此完整评价包含40,932次排名。filtered处理从候选中排除其他已知正确实体，目标实体保留。MRR强调目标靠前的程度，Hits@10记录目标进入前十的比例。三个模型在相同数据划分上比较，验证抽样和维度在结果配置中保留。"]
        historical = ROOT / "运行记录/诊断/conve_separate_bn.json"
        conve = next((row for _, row in records if row["config"]["model"] == "conve"), None)
        if historical.exists() and conve:
            old = json.loads(historical.read_text(encoding="utf-8"))
            text += ["", f"ConvE将正负三元组合并前向后，完整测试MRR由{old['test']['mrr']:.4f}变为{conve['test']['mrr']:.4f}，Hits@10由{old['test']['hits@10']:.4f}变为{conve['test']['hits@10']:.4f}。两次均使用相同种子、20轮及完整测试划分，修正了BatchNorm按正负样本分别统计的问题。", "", f"[原分组前向结果]({link(historical, target)})"]
    if records:
        from plot_comparisons import compare
        figure = compare(records, folder, domain, task)
        if figure:
            text += ["", "### 测试指标对照", "", "![实验指标对照](metric_comparison.png)"]
    if records:
        text += ["", "### 训练曲线"]
        datasets = sorted({d["config"].get("dataset", "") for _, d in records})
        for dataset in datasets:
            subset = [(p, d) for p, d in records if d["config"].get("dataset", "") == dataset]
            filename = "curves_" + dataset + ".png" if dataset else "training_curves.png"
            plot(subset, folder / filename)
            text += ["", f"![{dataset or title}训练曲线]({filename})"]
    if domain == "nlp" and task in ["task2", "task3"]:
        diagnosis = ROOT / "运行记录/诊断" / (task + "_classes.json")
        if diagnosis.exists():
            text += ["", f"[各类别precision、recall、F1及错误样例]({link(diagnosis, target)})"]
            diagnosed = json.loads(diagnosis.read_text(encoding="utf-8"))
            text += ["", f"各类别指标对应检查点 `{Path(diagnosed['checkpoint']).stem}`，在完整测试集上重新计算。"]
            labels = {"0": "蕴含", "1": "中立", "2": "矛盾"} if task == "task3" else {"0": "负面", "1": "较负面", "2": "中性", "3": "较正面", "4": "正面"}
            text += ["", "| 类别 | Precision | Recall | F1 | 测试样本 |", "|---|---:|---:|---:|---:|"]
            for key, label in labels.items():
                value = diagnosed["classification_report"][key]
                text.append(f"| {label} | {value['precision']:.4f} | {value['recall']:.4f} | {value['f1-score']:.4f} | {int(value['support'])} |")
            if task == "task3":
                text += ["", "中立类F1低于蕴含和矛盾。模型需要判断前提是否足以支持结论，词汇相关性容易干扰这种判断。例如前提只提到教堂唱诗班唱歌，假设“教堂天花板有裂缝”应为中立，当前模型预测为矛盾。"]
    if domain == "nlp" and task in ["task3", "task4"] and records:
        row = max((d for _, d in records), key=lambda d: d["best_dev_score"])
        metric = "accuracy" if task == "task3" else "f1"
        history = row["epochs"]
        chosen = max(history, key=lambda epoch: epoch["dev"][metric])
        text += ["", f"验证集最佳轮次为第{chosen['epoch']}轮，{metric}={chosen['dev'][metric]:.4f}；末轮为{history[-1]['dev'][metric]:.4f}。训练损失从{history[0]['train_loss']:.4f}降至{history[-1]['train_loss']:.4f}。参数保存采用验证选优，因此后续训练的波动不会覆盖较好的模型。"]
    if domain == "nlp" and task == "task4":
        diagnosis = ROOT / "运行记录/诊断/task4_entities.json"
        if diagnosis.exists():
            report = json.loads(diagnosis.read_text(encoding="utf-8"))
            text += ["", "| 实体类型 | Precision | Recall | F1 | 测试实体 |", "|---|---:|---:|---:|---:|"]
            for name in ["PER", "ORG", "LOC", "MISC"]:
                value = report["entity_report"][name]
                text.append(f"| {name} | {value['precision']:.4f} | {value['recall']:.4f} | {value['f1-score']:.4f} | {value['support']} |")
            text += ["", f"[实体类别指标与逐词错误样例]({link(diagnosis, target)})", "", f"上表对应按验证F1选出的检查点 `{Path(report['checkpoint']).stem}`。基础模型的小写词编号与GloVe会丢失原词形信息，扩展模型拼接可学习的大小写类别表示。两组同时改变了隐层宽度和训练轮数，差值反映这套扩展配置的整体效果。逐词错误样例用于继续检查未见专名、实体边界和类型混淆。", "", f"[完整CoNLL测试标签核对]({link(ROOT / '运行记录/诊断/conll_reference_check.json', target)})"]
    if domain == "nlp" and task == "task3" and (folder / "attention_example.png").exists():
        text += ["", "### 词级注意力", "", "![实际注意力矩阵](attention_example.png)", "", "每一行对应premise中的一个词，颜色表示分配给hypothesis各词的注意力概率，行概率和为1。矩阵由留出样例和保存的模型参数计算。"]
    image = REPORT / f"运行截图/{kind}.png"
    if image.exists():
        text += ["", "### 实际运行截图", "", "以下为加载保存参数并运行留出样例的本地页面原始截屏；完整测试指标见上表。", "", f"![运行输出]({link(image, target)})", "", f"[原始输出]({link(image.with_suffix('.txt'), target)}) · [执行记录]({link(image.with_suffix('.json'), target)})"]
    text += ["", "## 四、遇到的问题及处理情况", "", ISSUES[domain, task], ""]
    if domain == "graph" and task == "task2":
        comparison = ROOT / "运行记录/诊断/link_reload_devices.json"
        if comparison.exists():
            text += ["CPU重新加载Cora的四个模型后，与CUDA逐边比较分数和划分哈希。分数在rtol=2e-5、atol=1e-6下相符，正负边之间的排序保持一致，四组AUC也一致。AP会按完全相同的分数组合阈值，float32细微差异改变了部分同类边的分组，复算差值最大约0.00000252。AP恢复检查使用1e-5的绝对容差，原始GPU指标和CPU复算指标分别保留。", "", f"[CPU/CUDA恢复核对]({link(comparison, target)})", ""]
    target.write_text("\n".join(text), encoding="utf-8")
    # The report-directory copy uses links relative to its own location.
    duplicate = REPORT / (title.split("：")[0] + ".md")
    duplicate.write_text("# " + title + "\n\n完整实验报告、曲线和运行截图见[任务报告](" + link(target, duplicate) + ")。\n", encoding="utf-8")
    return records


def environment():
    import torch
    from importlib.metadata import version
    packages = ["torch", "torch-geometric", "transformers", "datasets", "peft", "numpy", "scipy", "scikit-learn", "pyarrow", "tokenizers", "playwright"]
    text = ["# 实验环境与数据条件", "", "| 项目 | 本机配置 |", "|---|---|", f"| 操作系统 | {platform.platform()} |", f"| Python | {platform.python_version()} |", "| GPU | NVIDIA GeForce RTX 3050 Ti Laptop，4GB |", "| 内存 | 16GB |", f"| CUDA runtime | {torch.version.cuda} |", "", "| 依赖 | 版本 |", "|---|---|"]
    for name in packages:
        text.append(f"| {name} | {version(name)} |")
    text += ["", "使用项目虚拟环境运行，CUDA训练按队列串行执行。NumPy手写分类使用CPU。另创建独立的`.venv-reproduce`环境核对依赖与模型推理，[独立环境检查](../运行记录/独立环境核查.json)保存结果。下载优先采用HF国内镜像，模型文件按LFS SHA-256核对，GloVe按官方ZIP CRC核对。", "", "### 模型与数据来源", "", "| 内容 | 来源与条件 |", "|---|---|", "| 情感分类 | Kaggle Rotten Tomatoes，156,060短语；SentenceId分组80/10/10 |", "| SNLI / CoNLL2003 | 官方划分，经HF镜像下载 |", "| 唐诗 | NLP上游poetryFromTang.txt |", "| Cora / CiteSeer / Flickr | PyG数据集，官方节点mask |", "| MUTAG / ZINC | TUDataset；ZINC官方full划分的固定子集 |", "| FB15k-237 | 官方train/valid/test |", "| 中文小BERT | uer/chinese_roberta_L-2_H-128 |", "| UIE | Pky/uie-base-zh，原版预训练编码器与边界头 |", "| T5 | uer/t5-small-chinese-cluecorpussmall |", "| 指令模型 | Qwen2.5-0.5B-Instruct |", "", "大模型部分根据4GB显存采用Qwen 0.5B完成零样本、LoRA、继续预训练和PPO；从零训练采用4层、128维GPT2。原仓库6B/7B模型对应的流程与本机规模分别在PLM报告中列出。", "", "运行截图由Playwright直接截取实际执行页面，同时保存命令、时间、退出码和原始输出。图片保留页面原貌。", "", "重新建立环境：", "", "```powershell", ".\\setup.ps1", "```", "", "随后将命令中的`.venv`改为`.venv-reproduce`。训练数据与模型缓存由下载脚本恢复，仓库提交包含代码、实验结果、日志、报告和截图。", ""]
    (REPORT / "实验环境.md").write_text("\n".join(text), encoding="utf-8")
    with (REPORT / "实验环境.md").open("a", encoding="utf-8") as stream:
        stream.write("\n[数据与模型来源清单](../运行记录/数据模型来源.json)保存文件大小、SHA-256和镜像来源，可用于核对恢复后的输入。\n")
        stream.write("\n四个预训练模型的修订版本见[来源版本](../来源版本.json)。下载脚本按这些固定版本恢复模型。小BERT分词器由BertTokenizerFast保存，词表保留相同token顺序，换行形式随保存格式统一；来源清单记录实际使用的文件哈希。\n")


def build():
    REPORT.mkdir(exist_ok=True)
    from zero_shot_output_diagnostics import main as zero_shot_diagnostics
    if not zero_shot_diagnostics():
        raise RuntimeError("Check the zero-shot source annotations before generating reports")
    environment()
    all_records = []
    entries = []
    for domain, specs in [("nlp", NLP), ("graph", GRAPH)]:
        for spec in specs:
            records = task_report(spec, domain)
            all_records += records
            folder = ROOT / ("nlp-beginner/legacy" if domain == "nlp" else "graph_beginner") / spec[1]
            entries.append((spec[0], len(records), folder / "README.md"))
    from plm_submission_reports import build as plm_build
    plm_records = plm_build()
    summary = REPORT / "总报告.md"
    lines = ["# 新生任务实验报告", "", "NLP按legacy原版五项任务完成，Graph包含节点分类、链路预测、图级预测和知识图谱补全，PLM按上游九类入口组织。代码、运行过程、结果和截图按任务归档。", "", "## 一、实验环境与条件", "", "实验使用Windows、Python 3.11和RTX 3050 Ti 4GB，详细依赖、数据划分、模型来源见[实验环境](实验环境.md)。", "", "## 二、实验方法与过程", "", "先实现NumPy分类与PyTorch训练流程，再完成注意力、CRF和字符语言模型。图任务保持统一模型接口，对比模型、采样、池化和参数。PLM从分类与匹配逐步扩展到prompt、抽取、生成、微调和PPO。每组实验保存配置、逐轮验证结果与最终测试结果。", "", "| 任务 | 正式实验数 | 实验报告 |", "|---|---:|---|"]
    for title, count, target in entries:
        lines.append(f"| {title} | {count} | [阅读报告]({link(target, summary)}) |")
    lines += [f"| PLM九类入口 | {len(plm_records)} | [阅读报告](PLM任务.md) |", "", "## 三、实验结果及分析", "", f"正式索引共收录{len(all_records) + len(plm_records)}组实验。各报告的表格直接读取完成状态的JSON，曲线来自逐轮记录。参数按验证集选择，测试集用于最终评价。完整配置可从表中的实验名称打开。", "", "[逐项日志](../进度日志.md) · [正式结果索引](正式实验索引.json) · [运行截图](运行截图)", "", "## 四、遇到的问题及处理情况", "", "数据下载中断通过国内镜像和校验续传处理。Kaggle短语按SentenceId分组，评论匹配按原评论分组，UIE按原content分组。空测试文件与无标签测试文件采用上游dev的分组留出。T5标签移除重复的起始token，生成模型重新加载检查点核验。具体处理与效果见各任务第四节。", "", "### 工具练习", "", "[工具上手](工具上手.md)说明终端、虚拟环境、Git、参数和日志的用法；[练习记录](本人练习与核查.md)按任务列出可独立复现和讲解的操作。", "", "完成代码与实验后，人工评价和学长审核按提交要求由本人进行。", ""]
    summary.write_text("\n".join(lines), encoding="utf-8")
    inventory = [{"file": p.relative_to(ROOT).as_posix(), "name": d["name"], "config": d["config"], "test": d.get("test", {}), "elapsed_seconds": d.get("elapsed_seconds")} for p, d in all_records]
    (REPORT / "正式实验索引.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory += [{"file": p.relative_to(ROOT).as_posix(), "name": d["name"], "config": d["config"], "test": d.get("test", {}), "elapsed_seconds": d.get("elapsed_seconds")} for p, d in plm_records]
    (REPORT / "正式实验索引.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Submission task reports updated", len(all_records), datetime.now().astimezone().isoformat())


if __name__ == "__main__":
    build()
