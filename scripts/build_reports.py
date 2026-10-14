"""Generate Chinese reports strictly from saved experimental outputs."""
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "学习报告"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def results(folder):
    values = []
    for path in sorted(folder.glob("*.json")):
        d = read(path)
        if d.get("status") != "completed":
            continue
        if path.stem in ["pointwise_n0_lr0.0003", "dssm_n0_lr0.0003", "sbert_n0_lr0.0003"] and path.with_name(path.stem + "_grouped.json").exists():
            continue
        if path.stem == "uie_negative" and path.with_name("uie_negative_v2.json").exists():
            continue
        arm = path.stem.replace("_v2", "")
        if arm in ["uie_none", "uie_negative", "uie_swap"] and path.with_name(arm + "_v3.json").exists():
            continue
        if path.stem == "qwen_zero_shot" and path.with_name("qwen_zero_shot_v2.json").exists():
            continue
        if path.stem == "qwen_ppo" and path.with_name("qwen_ppo_v2.json").exists():
            continue
        if path.stem in ["t5_qa_n1000", "t5_filling_n1000"] and path.with_name(path.stem + "_v2.json").exists():
            continue
        values.append((path, d))
    return values

def number(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

def compact(metrics):
    keep = ["accuracy", "macro_f1", "f1", "auc", "average_precision", "mae", "mrr", "hits@10", "loss", "perplexity", "preference_accuracy", "exact_match", "character_f1"]
    return "; ".join(k + "=" + number(metrics[k]) for k in keep if k in metrics) or "—"

def result_table(records):
    lines = ["| 实验 | 训练规模 | 轮数 | 验证集选优 | 测试结果 | 总耗时(s) |", "|---|---:|---:|---|---|---:|"]
    for path, d in records:
        c = d.get("config", {})
        counts = c.get("counts", d.get("counts", {}))
        count = counts.get("train", c.get("training_triples", c.get("nodes", "—")))
        best = "; ".join(k.replace("best_dev_", "") + "=" + number(v) for k, v in d.items() if k.startswith("best_dev_")) or "—"
        test = compact(d.get("test", {}))
        if "test_accuracy" in d:
            test = "accuracy=" + number(d["test_accuracy"]) + "; macro_f1=" + number(d["test_macro_f1"])
        if "test_perplexity" in d:
            test = "perplexity=" + number(d["test_perplexity"])
        lines.append(f"| [{d['name']}]({path.resolve().as_posix()}) | {count} | {len(d.get('epochs', []))} | {best} | {test} | {d.get('elapsed_seconds', 0):.1f} |")
    return "\n".join(lines)

def plot(records, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        count = 0
        selected = records if len(records) <= 12 else records[::math.ceil(len(records) / 8)]
        for _, d in selected:
            c = d.get("config", {})
            if c.get("features") in ["count", "binary", "tfidf"] and "ngram" in c:
                label = f"{c['features'][:2]} n{c['ngram']} {c['loss'][:2]} lr{c['lr']} b{c['batch']} sh{int(c['shuffle'])}"
            elif "dataset" in c:
                label = f"{c['dataset']} {c.get('model','')} {c.get('pool','')} {c.get('mode','')} l{c.get('layers','')} lr{c.get('lr','')} s{c.get('seed',42)}"
            else:
                label = d["name"]
            history = d.get("epochs", [])
            if not history:
                continue
            loss_key = next((k for k in ["train_loss", "loss"] if k in history[0]), None)
            if loss_key:
                axes[0].plot(range(1, len(history)+1), [e[loss_key] for e in history], label=label, alpha=.7)
            dev = history[0].get("dev", {})
            key = next((k for k in ["accuracy", "auc", "f1", "mae", "mrr", "preference_accuracy", "loss"] if k in dev), None)
            if key:
                axes[1].plot(range(1, len(history)+1), [e["dev"][key] for e in history], label=label + ":" + key, alpha=.7)
            elif "dev_accuracy" in history[0]:
                axes[1].plot(range(1, len(history)+1), [e["dev_accuracy"] for e in history], label=label)
            elif "dev_perplexity" in history[0]:
                axes[1].plot(range(1, len(history)+1), [e["dev_perplexity"] for e in history], label=label)
            count += 1
        for ax, title in zip(axes, ["Training loss", "Validation metric (see legend)"]):
            ax.set_title(title)
            ax.set_xlabel("Epoch / iteration")
            ax.grid(alpha=.25)
        if count <= 12:
            axes[1].legend(fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(target, dpi=150)
        plt.close(fig)
    except ImportError:
        pass

def observations(records):
    lines = ["## 本次结果怎么读", ""]
    valid = []
    for _, d in records:
        for k, v in d.items():
            if k.startswith("best_dev_") and isinstance(v, (float, int)):
                low = "loss" in k or (k == "best_dev_score" and d.get("config", {}).get("problem") == "regression")
                valid.append((d, k, v, low))
                break
    if valid:
        for dataset in sorted({d.get("config", {}).get("dataset", "当前数据") for d, *_ in valid}):
            group = [r for r in valid if r[0].get("config", {}).get("dataset", "当前数据") == dataset]
            best = min(group, key=lambda r: r[2]) if group[0][3] else max(group, key=lambda r: r[2])
            d, key, value, _ = best
            lines += [f"{dataset}中，按验证指标选优的记录是 `{d['name']}`，{key}={number(value)}。其测试结果：{compact(d.get('test', {})) if d.get('test') else '见上表'}。这只说明当前种子和当前配置，不能据此认定该模型在其他数据上也最好。", ""]
    pairs = {}
    for _, d in records:
        c = d.get("config", {})
        if c.get("mode") in ["full", "sampled", "batch"]:
            key = (c.get("dataset"), c.get("model"), c.get("pool", ""), c.get("layers"), c.get("lr"))
            pairs.setdefault(key, {})[c["mode"]] = d
    comparisons = []
    for key, values in pairs.items():
        other = "sampled" if "sampled" in values else "batch"
        if "full" not in values or other not in values:
            continue
        a, b = values["full"], values[other]
        at, bt = a.get("train_seconds"), b.get("train_seconds")
        if at is None or not bt:
            continue
        comparisons.append(f"| {'/'.join(str(k) for k in key[:3] if k)} | {at:.3f} | {bt:.3f} | {at / bt:.3f} | {compact(a.get('test', {}))} | {compact(b.get('test', {}))} |")
    if comparisons:
        lines += ["下面耗时只统计训练阶段，不包含验证、测试和下载。倍率=full训练耗时/另一模式训练耗时；超过1表示另一模式训练更快。分区丢边、监督量与每轮更新次数也会改变，不能据倍率单独下结论。", "", "| 数据/模型/池化 | full训练(s) | 另一模式训练(s) | 倍率 | full测试 | 另一模式测试 |", "|---|---:|---:|---:|---|---|", *comparisons, ""]
    glove = [(p, d) for p, d in records if d.get("glove_coverage") is not None]
    for _, d in glove:
        lines += [f"`{d['name']}` 的GloVe词表覆盖率为 {d['glove_coverage']:.2%}，余下词随机初始化。比较随机初始化时，应同时查看验证和测试指标；一个种子的差异不足以证明稳定提升。", ""]
    for _, d in records:
        if d.get("generated"):
            lines += [f"`{d['name']}` 实际生成：", "", "```json", json.dumps(d["generated"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)

def document(name, folder, knowledge, commands, notes, expected):
    folder.mkdir(exist_ok=True, parents=True)
    records = results(folder / "results")
    lines = [f"# {name}", "", "## 做了什么", "", notes, "", f"当前已完成 {len(records)} 组记录，计划覆盖 {expected} 组。表中结果来自本地 JSON；缺少的组不算完成。模型按验证集指标选优，再进行测试。", "", result_table(records), "", observations(records), "", "## 怎么运行", "", "在总目录的 PowerShell 中执行。首次使用先执行 `$env:PYTHONUTF8='1'`，以下命令均使用项目虚拟环境。", "", "```powershell", *[".\\.venv\\Scripts\\python.exe " + cmd for cmd in commands], "```", "", "## 知识点与实现", "", knowledge, "", "## 自己动手与核查", "", "1. 亲自运行一个配置，在 JSON 中找到配置、验证指标和测试指标。", "2. 修改一个参数并另存结果，解释预测或训练曲线为什么改变。", "3. 选一条样例，从输入张量形状一直讲到输出和损失，不能只背模型名字。", "", "模型检查点位于 `checkpoints`；训练输出位于总目录的 `运行记录`。自动执行记录用于辅助学习；交给学长前，需要本人完成上面的复现和解释。", ""]
    if records:
        plot(records, folder / "training_curves.png")
        if (folder / "training_curves.png").exists():
            lines += ["## 训练曲线", "", f"![训练曲线]({(folder / 'training_curves.png').as_posix()})", "", "图按每份记录的原始轮数作图。超过12组时等间隔展示约8组，全部原始曲线数据仍在JSON中。模型、数据集或指标不同的曲线不能直接当成同一实验的优劣比较。", ""]
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT / (name.split('：')[0] + ".md")).write_text("\n".join(lines), encoding="utf-8")
    return records

NLP = [
    ("NLP任务1：NumPy文本分类", "task1", 9,
     "词袋只统计单词出现次数，不记录顺序。binary 只保留是否出现；TF-IDF 降低常见词的权重；二元 n-gram 可表示局部搭配。词表与 IDF 只在训练集拟合，验证和测试只能 transform。\n\nSoftmax 的概率是 `exp(z_k)/sum(exp(z))`，交叉熵为 `-log p_y`。先减最大 logit 避免指数溢出。梯度是 `X.T @ (p-one_hot(y))/B`。本实现的 logistic 是五个 one-vs-rest 二分类器，BCE 对类别求和，再取样本均值。参数更新由 NumPy 手写，sklearn 只负责特征和评估。\n\n`batch=0` 表示全批次，`batch=1` 表示逐样本 SGD，`batch=64` 表示 mini-batch。同为20轮时，更新次数并不相同。shuffle 每轮重排样本；不shuffle可能使更新带上数据排序的偏差。学习率过大可能振荡，过小可能还未学够，需看验证曲线。",
     ["nlp-beginner/legacy/train_numpy.py --features tfidf --loss softmax --epochs 20", "scripts/run_matrix.py nlp"],
     "完成词袋/binary/TF-IDF、unigram/bigram、softmax/OVR logistic、学习率、batch和shuffle的9组对比。实际数据为 SetFit/sst5 官方句子级训练/验证/测试（8544/1101/2210），来源与 Rotten Tomatoes 相近；未使用 Kaggle 全短语 train.tsv。`prepare_data.py sentiment --kaggle-tsv 路径` 支持按 SentenceId 分组读取 Kaggle 数据。"),
    ("NLP任务2：PyTorch分类与词嵌入", "task2", 7,
     "`train_torch_bow.py` 用相同 TF-IDF、零初始化和 SGD 复写任务1，并检查两种损失及参数梯度与 NumPy 一致。另一个名为 softmax 的模型是词向量均值加线性分类器，它与 TF-IDF 基线的表示不同，应分开理解。\n\nCNN 用3/5/7宽度卷积提取局部特征，ReLU后进行最大池化；padding位置不能参与池化。RNN为双向LSTM，按真实长度打包，再拼接均值和最大池化特征。Embedding 把离散词编号映射为可训练向量。GloVe初始化有语料知识，未命中的词仍随机初始化，报告里保留覆盖率，训练时不冻结词向量。\n\nDropout 在训练时随机置零部分特征，推理必须 `model.eval()`。`zero_grad → forward → backward → step` 是基本训练顺序；验证用 `no_grad`。",
     ["nlp-beginner/legacy/train_torch_bow.py --loss softmax", "nlp-beginner/legacy/train_torch_bow.py --loss logistic", "nlp-beginner/legacy/train_torch.py classification --model cnn --epochs 8", "nlp-beginner/legacy/train_torch.py classification --model cnn --epochs 8 --glove nlp-beginner/legacy/data/raw/glove/glove.6B.50d.txt"],
     "PyTorch TF-IDF两种损失；均值词向量分类、CNN、BiLSTM的随机初始化；CNN/BiLSTM的GloVe50维初始化，共7组。GloVe从官方ZIP取50维条目，按原始大小与CRC验证。与NumPy分类共用句子级SST数据。"),
    ("NLP任务3：ESIM自然语言推断", "task3", 1,
     "先用共享BiLSTM编码 premise 与 hypothesis。对任意两个词的隐状态计算点积 `e_ij = a_i dot b_j`；在对方有效token上softmax，得到两向对齐表示。拼接 `x, aligned, x-aligned, x*aligned`，经投影和第二个BiLSTM组合，最后对两句做均值/最大池化，预测蕴含、矛盾或中立。\n\n注意力softmax前要屏蔽padding，否则模型会把概率分给不存在的词。SNLI中label=-1的无共识样本被排除。3分类的随机水平约1/3，不能把高于随机水平就解释为模型完全理解逻辑。",
     ["nlp-beginner/legacy/prepare_data.py snli", "nlp-beginner/legacy/train_torch.py matching --model esim --limit 20000 --epochs 5"],
     "SNLI官方划分已完整下载。训练使用固定种子的2万条子集，验证和测试各1000条，5轮；这是有限资源入门实验，不是全量SNLI复现。"),
    ("NLP任务4：BiLSTM-CRF实体识别", "task4", 1,
     "BiLSTM给出每个位置每种BIO标签的发射分数，CRF再加相邻标签的转移分数及起止分数。一条标签路径的分数为这些项之和。负对数似然是 `logsumexp(所有路径分数)-正确路径分数`。前向动态规划用logsumexp计算配分函数，Viterbi把logsumexp换为max并记录回溯指针。\n\n实体级precision/recall/F1要求实体边界和类型都正确，不能用token accuracy替代。长短句共批训练时，mask必须同时进入路径评分、配分函数和解码。已用短序列穷举所有路径核对动态规划和Viterbi。当前是普通可学习转移CRF，没有额外强制BIO转移约束。",
     ["nlp-beginner/legacy/train_torch.py ner --model bilstm-crf --epochs 8", "scripts/verify_algorithms.py"],
     "使用CoNLL-2003官方训练/验证/测试，全量训练8轮，最大句长100。超过100个token的尾部在本实验中被截断；实体F1来自截断后序列。保存了自写CRF和完整检查点。"),
    ("NLP任务5：唐诗字符语言模型", "task5", 2,
     "字符语言模型预测下一个字符，训练输入与标签错开一个位置。LSTM有输入、遗忘、输出门和独立cell state；GRU用更新与重置门合并部分状态。交叉熵按有效字符平均，困惑度为 `exp(平均负对数似然)`，越低表示模型对真实下一个字符越有把握。\n\n不能先对重叠窗口随机分割数据，否则训练与测试可能看到同一段文字。这里先按连续文本80/10/10划分，再分别建立窗口。生成用前文状态循环采样，温度0.7更集中、1.0更分散。词表只由训练区段建立，测试未见字符映射到unk并记数量。",
     ["nlp-beginner/legacy/train_torch.py lm --model lstm --epochs 20", "nlp-beginner/legacy/train_torch.py lm --model gru --epochs 20"],
     "使用仓库poetryFromTang.txt，字符级LSTM和GRU各20轮。记录验证和测试困惑度、连续区段字符数、未见字符数，以及两种温度下实际生成的文本。")]

GRAPH = [
    ("Graph任务1：节点分类", "task1", 28,
     "GCN对邻居特征做含度归一化的传播；GAT学习邻居注意力；GraphSAGE分别处理自身与邻居聚合；GIN用求和聚合加MLP，并学习自身项的系数。四者共享层数、隐藏维度与训练流程，便于观察差异。\n\nCora/CiteSeer采用公开mask；Flickr保留官方role划分。节点分类是transductive设置：训练可以使用全图无标签结构，但损失只能用train_mask对应标签。验证集选择轮次，测试集最后评估。\n\nRandomNodeLoader是框架的随机诱导子图分区。跨分区边会丢失，不能当成多跳邻居采样。分区降低单次显存占用，但可能损失邻居信息。层数增加也可能出现过平滑：不同节点表示逐渐接近。",
     ["graph_beginner/task1/code/train.py --dataset Cora --model gcn --mode full --epochs 100", "scripts/run_matrix.py graph-small", "scripts/run_matrix.py graph-flickr"],
     "三数据集、四模型、全图/采样共24组；Cora GCN另有4组学习率/层数消融，共28组。Flickr隐藏维度16，小图32。训练时间在CUDA同步后记录；早期小图实验与其他训练有并发，因此时间只作粗略参考，不作为严格独占GPU基准。"),
    ("Graph任务2：链路预测", "task2", 24,
     "编码器产生节点向量z，边分数为 `z_u dot z_v`，二元交叉熵区分真实边和负样本。ROC-AUC评价排序区分能力，AP更关注正例召回精度。负例比例会影响AP，比较时应保留相同设置。\n\nRandomLinkSplit把边分为训练、验证和测试，20%的训练正边额外与消息传递边分离。训练图中没有验证/测试正边；验证只用允许的训练结构，测试按框架标准可利用训练及验证消息边。负例由框架从所有正边之外取，避免把留出正边错当负例。\n\n采样时把全局节点ID映射到局部ID，只保留两端都在当前分区的监督边。因而采样训练每轮监督边数减少；JSON记录了数量。这种对比同时改变结构与监督量，不能只把结果归因于运行方式。",
     ["graph_beginner/task2/code/train.py --dataset Cora --model sage --mode sampled --epochs 50", "scripts/run_matrix.py graph-flickr"],
     "Cora/CiteSeer/Flickr、GCN/GAT/GraphSAGE/GIN、full/sampled，共24组。正负边约1:1，10%验证、10%测试，固定种子，50轮。"),
    ("Graph任务3：图级分类与回归", "task3", 48,
     "节点表示经过池化得到整张图的表示。mean比较平均特征，max取各通道最强响应，min取各通道最低响应。必须用batch向量区分不同图，不能将一个批次所有节点池成一张图。\n\nMUTAG是图分类，使用cross entropy、accuracy和macro-F1。ZINC预测分子连续性质，使用L1损失及MAE，不可计算分类准确率。其原子编号先查embedding；当前四种基本编码器使用原子类别与图连接，未使用键类型特征，属于简单基线。\n\nfull每轮只做一次全部训练图的更新，batch每轮做多次更新，因此同轮数的性能差异也包含优化步数差异。MUTAG测试仅19张图，单个样本就明显改变accuracy，结论需要谨慎。",
     ["graph_beginner/task3/code/train.py --dataset MUTAG --model gin --pool mean --mode batch", "scripts/run_matrix.py graph-zinc"],
     "MUTAG使用PyG TUDataset，按固定种子80/10/10分图。ZINC来自graphs-datasets/ZINC的官方划分镜像，固定采样训练2000、验证1000、测试1000；它不是PyG预定义的10k subset。两数据集×四模型×mean/max/min×full/batch，共48组，30轮。"),
    ("Graph任务4：知识图谱补全", "task4", 3,
     "TransE希望 `h+r≈t`，以负距离评分；RotatE把实体放在复数空间，用单位模关系向量旋转头实体；ConvE把头实体与关系重排成二维网格，经卷积、投影，与尾实体向量相乘评分。负采样随机替换头或尾，剔除训练已知真三元组。\n\n评估为每条三元组分别替换头和尾，在全部实体中排序。filtered评估先排除其余已知真三元组，再保留当前正确实体，避免另一个真实答案被算成错误。MRR是倒数排名的平均值，Hits@k是正确实体位于前k名的比例。本实现用optimistic ties。",
     ["scripts/run_matrix.py kge", "graph_beginner/task4/code/train.py --model rotate --limit 20000 --epochs 5 --eval-limit 50"],
     "FB15k-237官方划分，训练2万条子集、5轮。验证与测试各固定取50条，头尾都在完整实体候选集中filtered排序。50条评估波动很大，不能作为论文水平结论。")]

PLM_KNOWLEDGE = """## 1. 分类与文本匹配

BERT分类取CLS表示接分类头。Pointwise把两句拼到同一编码器，可直接建模交互。DSSM分别编码两句，通过投影后的余弦分数学习匹配，适合离线建库。SBERT的监督分类头输入 `u,v,abs(u-v)`。SimCSE用同一句话的两次独立Dropout作为正对，同批其他句子作为负对，以温度0.05计算InfoNCE；训练不使用语义标签。

监督匹配数据中同一评论会与多个类别组成不同句对，最终报告使用按评论分组的dev/test，避免同一评论跨两边。SimCSE的train.txt是无标签单句，test.tsv是无标签句对。带标签dev按共享句子的连通组划为验证/测试；验证选择相似度阈值，测试固定阈值。小型预训练BERT是uer/chinese_roberta_L-2_H-128。

## 2. UIE与增强

这里用小BERT编码prompt+content，用两个边界头预测起点和终点，是UIE风格实现，未加载UIE-base预训练权重。损失只算content有效token，屏蔽prompt、特殊token和padding。原test为空，按原content分组把dev一半留作test，并移除训练重叠content。AutoNeg v2使用baseline在**已标注训练负例**中的误报进行重复训练；AutoPos/Swap在同一prompt下替换单实体，并更新偏移。增强数据只从训练集生成。

## 3. PET与p-tuning

PET把类别变成固定标签词，在人工模板中的MASK位置预测标签。p-tuning冻结BERT权重，只训练插在CLS后的连续prompt向量。类别schema含上游dev中出现而train未出现的“电器”，因此该类别没有训练示例。标签schema只是类别定义，不能用测试样本更新模型。

## 4. T5问答与填空

T5是encoder-decoder，编码问题与原文，再自回归生成答案。训练用teacher forcing；padding标签改为-100，避免模型学着生成padding。BERT tokenizer的token_type_ids要去掉，T5不接受此参数。用验证loss选最佳模型；测试实际generate后计算去空白的exact match和字符重叠F1，字符F1仅为本次中文简易指标。问答按context分组、填空按target分组留出数据。使用中文T5-small，训练1000条子集，不代表完整QA系统。

## 5. 零样本与MBTI

零样本提示里只给任务说明和类别，没有示例。分类、匹配各评估20条留出样例，抽取20条，保存原始回答与解析成功率；样本少，不宜泛化结论。MBTI逐题要求A/B，以题目的维度映射计票；无有效A/B不计票，平票记问号。结果只是这个模型在该提示下的输出倾向，不能认定模型具有人类人格。

## 6. LoRA、继续预训练与从零训练

本机4GB显存，ChatGLM-6B与原7B训练配置没有按原权重运行。可运行替代为Qwen2.5-0.5B-Instruct，BF16、q_proj/v_proj的rank8 LoRA。LoRA只训练低秩增量，冻结原权重。SFT只对assistant目标计算损失，prompt及padding标签为-100；prompt保留尾部最多半个窗口，target占剩余窗口。继续预训练对所有非padding字符计算下一token损失。两者更新目标不同。

从零训练使用只在训练文档拟合的3000词Byte-BPE和4层、128维GPT2。先学习语言模型，再用ShareGPT对话做assistant-only SFT。数据很少，生成质量可能低；这组用于理解训练链路，不能称为有用的大模型。从零小模型、预训练Qwen的loss也不能直接比较，tokenizer不同。

## 7. 奖励模型与PPO

排序评论由上游真实数据提供，训练目标为 `-log sigmoid(r(chosen)-r(rejected))`。把排名列表展开为所有有序偏好对，并保留独立测试。奖励分数是模型的相对标量，没有天然的绝对质量单位。

PPO先实际生成响应，保存旧策略每个响应token的log probability；通过训练好的奖励模型取得末端奖励，并逐token施加KL约束。优势用GAE计算，策略比率为 `exp(new_logp-old_logp)`，目标为 `min(ratio*A,clip(ratio)*A)`；价值头拟合回报并做value clipping。参考策略是关闭LoRA时的固定Qwen权重。此处自动奖励来自偏好模型，不是刚由本人给出的人工反馈。人类评分工具已提供，等本人读样例后填写。

## 8. 分词工具

`local_tools.py`只监听127.0.0.1，可显示token字符串、ID、字符偏移和decode，并比较中文BERT、Qwen及自己训练的BPE。中文BERT多按字分词，Byte-BPE的可见token可能像乱码，decode后才能恢复原文本。运行：

```powershell
.\\.venv\\Scripts\\python.exe transformers_tasks/local_runs/local_tools.py
```

浏览器打开 http://127.0.0.1:8765 。人工评分不会预填；本人选分并提交后才追加到human_ratings.jsonl。
"""

def build():
    REPORT.mkdir(exist_ok=True)
    all_records = []
    for name, task, expected, knowledge, commands, notes in NLP:
        all_records += document(name, ROOT / "nlp-beginner/legacy" / task, knowledge, commands, notes, expected)
    for name, task, expected, knowledge, commands, notes in GRAPH:
        all_records += document(name, ROOT / "graph_beginner" / task, knowledge, commands, notes, expected)
    folder = ROOT / "transformers_tasks/local_runs"
    plm = results(folder / "results")
    all_records += plm
    plm_text = "# PLM任务运行与学习报告\n\n上游300个文件已核对Git blob哈希，源码保留。新实现均在local_runs；本报告不把小模型替代写成原6B/7B实验完成。\n\n" + result_table(plm) + "\n\n" + PLM_KNOWLEDGE
    special = ["\n## 应用与强化学习的实际结果\n"]
    for _, d in plm:
        if d.get("tasks"):
            for task, metrics in d["tasks"].items():
                score = {k: v for k, v in metrics.items() if k != "examples"}
                special += [f"零样本 {task}：`{json.dumps(score, ensure_ascii=False)}`。", "", "```json", json.dumps(metrics["examples"][:1], ensure_ascii=False, indent=2), "```", ""]
        if "profile" in d:
            special += [f"MBTI输出倾向：{d['profile']}；有效回答 {d['valid_responses']}/{d['total_questions']}，各维度计票 `{json.dumps(d['votes'], ensure_ascii=False)}`。这不是人格诊断。", ""]
        if d.get("samples"):
            history = d["epochs"]
            special += [f"PPO共{len(history)}轮、{len(d['samples'])}条真实生成。第一轮平均自动奖励 {history[0]['reward']:.4f}，最后一轮 {history[-1]['reward']:.4f}；采样波动和短训练都很明显，不能仅凭末轮数值认定持续改善。sampled_kl是带符号的蒙特卡洛估计，单批可能为负；它不等于计算出负的理论KL。", "", "```json", json.dumps(d["samples"][:2], ensure_ascii=False, indent=2), "```", ""]
    plm_text += "\n".join(special)
    plm_text += "\nT5最终复现命令（修正目标中重复CLS后，使用验证集观察到的训练改善加长训练）：\n\n```powershell\n.\\.venv\\Scripts\\python.exe transformers_tasks/local_runs/seq2seq_tasks.py qa --epochs 10 --batch 8 --lr .0005\n.\\.venv\\Scripts\\python.exe transformers_tasks/local_runs/seq2seq_tasks.py filling --epochs 10 --batch 8 --lr .0005\n```\n"
    for _, d in plm:
        if d.get("config", {}).get("parsing"):
            plm_text += "\n填空数据过滤统计（缺少制表符或空输入/目标）：\n\n```json\n" + json.dumps(d["config"]["parsing"], ensure_ascii=False, indent=2) + "\n```\n"
    plm_text += "\n## 复现命令\n\n```powershell\n" + "\n".join(".\\.venv\\Scripts\\python.exe " + command for command in ["transformers_tasks/local_runs/bert_tasks.py classification", "transformers_tasks/local_runs/bert_tasks.py pointwise", "transformers_tasks/local_runs/bert_tasks.py dssm", "transformers_tasks/local_runs/bert_tasks.py sbert", "transformers_tasks/local_runs/bert_tasks.py simcse --limit 10000 --epochs 5", "transformers_tasks/local_runs/bert_tasks.py pet", "transformers_tasks/local_runs/bert_tasks.py ptuning --lr .01", "transformers_tasks/local_runs/span_task.py --augment none", "transformers_tasks/local_runs/span_task.py --augment negative", "transformers_tasks/local_runs/span_task.py --augment swap", "transformers_tasks/local_runs/reward_task.py", "transformers_tasks/local_runs/seq2seq_tasks.py qa", "transformers_tasks/local_runs/seq2seq_tasks.py filling", "transformers_tasks/local_runs/llm_tasks.py zero-shot", "transformers_tasks/local_runs/llm_tasks.py mbti", "transformers_tasks/local_runs/llm_tasks.py sft", "transformers_tasks/local_runs/llm_tasks.py cpt --epochs 1", "transformers_tasks/local_runs/llm_tasks.py scratch --limit 1000 --epochs 5 --batch 8 --accumulate 1 --lr .0005", "transformers_tasks/local_runs/llm_tasks.py scratch-sft --batch 8 --epochs 3 --accumulate 1", "transformers_tasks/local_runs/ppo_task.py"]) + "\n```\n"
    examples = []
    for _, d in plm:
        if d.get("examples"):
            examples += ["\n### " + d["name"], "\n```json", json.dumps(d["examples"][:2], ensure_ascii=False, indent=2), "```"]
    plm_text += "\n## 实际输出样例\n" + "\n".join(examples) + "\n"
    (folder / "README.md").write_text(plm_text, encoding="utf-8")
    (REPORT / "PLM任务.md").write_text(plm_text, encoding="utf-8")
    for title, prefixes in [("文本匹配", ["pointwise", "dssm", "sbert", "simcse"]), ("信息抽取", ["uie"]), ("Prompt", ["pet", "ptuning"]), ("分类", ["classification"]), ("RLHF", ["reward", "qwen_ppo"]), ("T5生成", ["t5_"]), ("大模型应用", ["qwen_zero_shot", "qwen_mbti"]), ("大模型训练", ["sft", "cpt", "scratch"])]:
        records = [(p, d) for p, d in plm if any(d["name"].startswith(s) for s in prefixes)]
        (REPORT / ("PLM_" + title + ".md")).write_text("# " + title + "\n\n" + result_table(records) + "\n\n完整流程和知识点见 [PLM任务](PLM任务.md)。\n", encoding="utf-8")
    stamp = datetime.now().astimezone().isoformat()
    summary = ["# 新生任务总报告", "", f"更新：{stamp}。共 {len(all_records)} 份已完成实验记录。", "", "## 完成范围", "", "三个仓库已拉到本地。NLP按本人确认的legacy五项任务；Graph按四项；PLM按仓库九类入口。结果、代码、日志和检查点均保存在本地。下面按实际记录检查覆盖，空缺会显示出来。", "", "| 项目 | 已完成组数 | 计划组数 | 报告 |", "|---|---:|---:|---|"]
    for name, task, expected, *_ in NLP:
        count = len(results(ROOT / "nlp-beginner/legacy" / task / "results"))
        summary.append(f"| {name} | {count} | {expected} | [{task}](../nlp-beginner/legacy/{task}/README.md) |")
    for name, task, expected, *_ in GRAPH:
        count = len(results(ROOT / "graph_beginner" / task / "results"))
        summary.append(f"| {name} | {count} | {expected} | [{task}](../graph_beginner/{task}/README.md) |")
    summary += ["", "## PLM覆盖状态", "", "| 入口 | 本地实验完成状态 |", "|---|---|"]
    expected_names = ["classification_n0_lr0.0003", "pointwise_n0_lr0.0003_grouped", "dssm_n0_lr0.0003_grouped", "sbert_n0_lr0.0003_grouped", "simcse_n10000_lr0.0003", "uie_none_v3", "uie_negative_v3", "uie_swap_v3", "pet_n0_lr0.0003", "ptuning_n0_lr0.01", "reward_n1000", "qwen_ppo_v2", "t5_qa_n1000_v2", "t5_filling_n1000_v2", "qwen_zero_shot_v2", "qwen_mbti", "sft", "cpt", "scratch", "scratch-sft"]
    completed = {d["name"] for _, d in plm}
    for name in expected_names:
        summary.append(f"| {name} | {'已运行，结果见PLM报告' if name in completed else '尚未完成'} |")
    summary += ["", "## 数据与资源限制", "", "- 情感分类用句子级SST，未复现Kaggle完整短语数据；脚本已支持本人提供train.tsv后按SentenceId分组。", "- ESIM训练2万；ZINC训练2000；KGE训练2万及50条测试三元组；T5训练1000；LoRA训练400。详细划分见各报告与JSON配置。", "- ChatGLM-6B/7B原权重与多卡配置未执行，用Qwen0.5B和从零小GPT2验证流程。UIE也使用小BERT边界模型，不能称为UIE-base结果。", "- 小图训练早期存在与其他任务并发，训练时间只作运行记录；采样与全图监督量不同；full/batch优化步数也不同。", "- 人工反馈、本人独立复现和学长核实仍须本人参与。自动生成的实验记录不等于已经过关。", "", "## 学习顺序", "", "先运行NumPy分类并手算一批梯度，再用PyTorch复写；接着讲清padding/mask、CNN/RNN、注意力与CRF。图任务从GCN开始，学会mask与边划分，再比较采样和池化。PLM从BERT微调转向prompt与生成，最后理解LoRA、奖励模型及PPO。每次修改一个参数，并把观察写进自己的记录。", "", "## 排错记录", "", "1. PyTorch原先是CPU版，项目虚拟环境改用CUDA12.4版，已能使用RTX3050Ti4GB。", "2. 网络下载超时和连接重置，用验证Content-Range的分块续传；模型及HF数据核对发布的SHA256，GloVe核对ZIP CRC。", "3. SimCSE文件格式不同：训练单句、dev带标签、test无标签。改按格式读取并另留独立带标签测试。", "4. PET dev出现未见类别，补完整类别schema并注明；UIE原test为空，按content分组留出。", "5. 监督匹配一条评论对应多个句对，修正为按评论分组dev/test，最终只采用grouped记录。", "6. NumPy词表编号是np.int32，序列化前转int；PyArrow21与NumPy1.26兼容。", "", "## 复现入口", "", "```powershell", "$env:PYTHONUTF8='1'", ".\\.venv\\Scripts\\python.exe scripts/verify_algorithms.py", ".\\.venv\\Scripts\\python.exe scripts/verify_plm_math.py", ".\\.venv\\Scripts\\python.exe scripts/remaining_runs.py", ".\\.venv\\Scripts\\python.exe scripts/additional_runs.py", ".\\.venv\\Scripts\\python.exe scripts/build_reports.py", "```", "", "remaining_runs按顺序运行剩余模型并跳过已完成结果，避免重复占显存。每个任务报告中有单项命令。依赖版本记录在requirements-lock.txt。"]
    summary += ["", "## 核查与本人上手", "", "[完成核查](../运行记录/完成核查.json)记录计划覆盖；[模型与数据审计](../运行记录/核查/audit.json)、[生成检查点重载](../运行记录/核查/generation_reload_checks.json)、[本地工具检查](../运行记录/local_tools_checks.json)及[情感数据来源复核](../运行记录/sentiment_source_check.json)保存实际核查证据。", "", "先读[工具上手](工具上手.md)，再按[本人练习与核查](本人练习与核查.md)亲自复现。人工评分和学长确认尚未代办。", "", "最终UIE三组在增强/挖掘完成后重设随机种子，使新模型从相同初始化开始；报告只采用v3对比。最终T5标签排除重复CLS，采用v2记录；初版失败输出保留在历史JSON中。"]
    (REPORT / "总报告.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (ROOT / "README.md").write_text("# 新生任务本地工作区\n\n[总报告](学习报告/总报告.md)记录完成状态、范围和复现入口。\n\n- [NLP五项任务](nlp-beginner/legacy/README.md)\n- [Graph四项任务](graph_beginner/README.md)\n- [PLM运行与学习报告](transformers_tasks/local_runs/README.md)\n- [逐项工作日志](进度日志.md)\n\n源码保留在三个仓库中，新实现分别放在legacy、task*/code及local_runs。所有分数来自实际实验输出。\n", encoding="utf-8")
    with (ROOT / "README.md").open("a", encoding="utf-8") as f:
        f.write("\n[工具上手](学习报告/工具上手.md) · [本人练习与核查](学习报告/本人练习与核查.md) · [完成核查](运行记录/完成核查.json)\n")
    inventory = [{"file": str(p.relative_to(ROOT)), "name": d["name"], "status": d["status"], "config": d["config"], "test": d.get("test", {}), "elapsed_seconds": d.get("elapsed_seconds")} for p, d in all_records]
    (REPORT / "实验索引.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    from importlib.metadata import version, PackageNotFoundError, distributions
    needed = ["torch", "torch-geometric", "transformers", "datasets", "accelerate", "peft", "numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "matplotlib", "seqeval", "tokenizers", "safetensors", "huggingface-hub", "tqdm", "requests", "joblib", "sentencepiece"]
    pins = ["--extra-index-url https://download.pytorch.org/whl/cu124"]
    for name in needed:
        try:
            pins.append(name + "==" + version(name))
        except PackageNotFoundError:
            pass
    (ROOT / "requirements-lock.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    metadata = sorted({(d.metadata.get("Name", ""), d.version) for d in distributions()})
    (REPORT / "环境包快照.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "进度日志.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp}：实验与报告快照\n\n生成 {len(all_records)} 份已完成结果索引，更新各任务README、知识点、复现命令与曲线。以下为全部已完成记录，具体分数见对应JSON。\n\n")
        for path, d in all_records:
            f.write(f"- {d['name']}：{len(d.get('epochs', []))}轮，{compact(d.get('test', {}))}，结果 `{path.relative_to(ROOT).as_posix()}`。\n")
    print("reports updated", len(all_records), stamp, flush=True)

if __name__ == "__main__":
    from submission_reports import build as submission_build
    submission_build()
