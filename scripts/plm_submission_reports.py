"""Write the nine PLM report sections from the corresponding saved experiments."""
import json
from pathlib import Path
from submission_reports import ROOT, REPORT, link, table, analyses
from build_reports import results, plot

GROUPS = [
    ("文本分类", ["classification"], ["plm-bert"],
     "使用中文小BERT的CLS表示接线性分类头，输入为上游评论分类数据，交叉熵训练20轮。验证集按macro-F1选优。",
     "CLS向量压缩了整句信息，分类头输出每个类别的logit。训练时backward计算梯度，AdamW更新参数；验证和预测使用eval模式。类别名称与上游label_mapping分别保存，方便查看数值标签与实际类别的对应关系。",
     "Windows下分类标签显式转换为LongTensor。训练集与留出集使用固定划分，类别schema在配置中保存。",
     ["bert_tasks.py classification --epochs 20"]),
    ("文本匹配", ["pointwise", "dssm", "sbert", "simcse"], ["plm-bert"],
     "监督任务比较Pointwise、DSSM和Sentence-BERT。Pointwise将两句共同编码，双塔模型分别编码两句。上游评论与多个类别组成句对，dev/test按原评论分组。无监督SimCSE使用单句训练，每句话独立经过两次Dropout，同批其他句子作为负例。",
     "单塔适合精细判断文本交互；双塔可先计算并缓存候选向量，便于检索。SBERT分类头拼接u、v与abs(u-v)。SimCSE的InfoNCE将正对相似度与整批相似度比较，温度为0.05，训练阶段只用单句和Dropout视图。相似度阈值在验证集选择，测试时固定。",
     "LCQMC训练文件是单句，dev含标签，test没有标签。按照文件实际格式读取，将有标签dev按共享句子的连通组划成验证与测试，并清除训练文本重叠。",
     ["bert_tasks.py pointwise", "bert_tasks.py dssm", "bert_tasks.py sbert", "bert_tasks.py simcse --limit 10000 --epochs 5"]),
    ("信息抽取", ["uie_"], ["plm-uie"],
     "先以小BERT和边界头训练条件抽取，再加载上游Pky/uie-base-zh的预训练编码器与起止头进行对照。预训练UIE比较原始训练、已标注训练负例中的误报挖掘、SwapSPO及Mask Then Fill四组。SwapSPO按相同谓词分组，将一条样例的主体和客体共同放入另一条样例的上下文。Mask Then Fill调用5,000条训练的T5填空模型，保护所有已标注片段，替换其他片段并同步修正实体偏移；掩码左右各保留60个字符，保证输入中能看到掩码。候选按长度、重复片段和基础UIE抽取一致性筛选。",
     "输入由prompt和content组成，编码上限256 token。起点与终点用两个二分类头预测，损失屏蔽prompt、特殊符号和padding，只在content有效token上计算，正边界权重为10。字符标注通过offset_mapping转换为token位置，预测token再还原为字符区间，跨度最多40 token。阈值从0.3、0.5、0.7中按验证F1选择。增强数据只由训练记录生成，各组从相同预训练权重和随机种子开始。预训练组使用batch8、lr=2e-5训练8轮。",
     "上游test为空，按content将dev分为验证和测试，并移除训练重复内容。UIE原始权重保存为旧版整模型对象，核对SHA-256后用受限读取器提取张量，转换为safetensors和现代配置。早期填空会出现重复人名、出生地等片段，长文本截断也可能丢失掩码；扩大填空训练后，改为掩码附近的局部上下文，限制重复生成，并用基础抽取模型检查增强前后的预测实体集合。增强样例的实体文本与偏移逐条核对。",
     ["span_task.py --pretrained-uie --augment none --epochs 8 --batch 8 --lr .00002", "span_task.py --pretrained-uie --augment negative --epochs 8 --batch 8 --lr .00002", "span_task.py --pretrained-uie --augment swap --epochs 8 --batch 8 --lr .00002", "span_task.py --pretrained-uie --augment fill --epochs 8 --batch 8 --lr .00002 --filling-checkpoint t5_filling_expanded"]),
    ("Prompt", ["pet", "ptuning"], ["plm-bert"],
     "在少量评论上比较PET与p-tuning。PET使用“这是一条[MASK][MASK]评论：…”模板，将类别映射到两个字的标签词。p-tuning在首个token后插入8个可学习连续向量，冻结BERT参数，仅训练prompt。",
     "PET把分类改写为掩码词预测，类别得分由标签词各位置的log概率求和。p-tuning中的prompt是可学习向量，优化时梯度经过冻结模型回传到这些向量。标签词长度保持相同，避免类别因token数量不同而产生不公平得分。",
     "原训练样例仅覆盖部分类别，dev出现“电器”。完整schema保留该类别，结果按真实留出标签评价。离散模板和连续prompt共用模型与划分，记录各自可训练参数。",
     ["bert_tasks.py pet --epochs 30", "bert_tasks.py ptuning --epochs 30 --lr .01"]),
    ("RLHF", ["reward", "qwen_ppo"], ["plm-reward", "plm-ppo"],
     "把上游排序评论展开为有序偏好对，训练BERT奖励模型。随后在Qwen 0.5B上进行12轮token级PPO更新，保存48条实际生成。每次使用4条训练提示，响应最多32 token，同一轮rollout更新2遍；学习率1e-5、clip=0.2、KL系数0.05。策略使用LoRA，另设价值头；固定参考模型为关闭LoRA后的基座。",
     "奖励损失为-log sigmoid(r(chosen)-r(rejected))。PPO保存旧策略响应token的log概率，计算exp(new_logp-old_logp)比率，以clip约束策略更新。GAE计算优势，价值头拟合回报；奖励中的KL项约束策略偏离参考模型。生成使用温度1、top_p=1、top_k=0，使采样分布与计算log概率一致。",
     "上游RLHF以GPT2进行PPO，本机使用Qwen 0.5B与LoRA连接同一个训练完成的奖励模型，保留token级策略更新、参考模型和价值头流程。奖励数据与训练文本分开管理，留出偏好对用于准确率评价。PPO的采样、clip与GAE通过数值检查。人工评价页面保留实际生成内容，评分由阅读者填写，记录在human_ratings.jsonl。",
     ["reward_task.py --limit 1000 --epochs 5", "ppo_task.py", "local_tools.py"]),
    ("T5生成", ["t5_"], ["plm-qa", "plm-filling"],
     "用中文T5-small完成问答与填空。保留1,000条训练的基线，新增5,000条训练、各500条验证与测试的实验。问答按context分组，填空按目标文本分组留出。输入上限192 token，目标上限64 token；训练采用teacher forcing，测试实际生成文本，计算完全匹配率与字符F1。",
     "T5编码器读取问题和上下文，解码器逐token生成答案。输入中的token_type_ids移除，标签padding设为-100。decoder_start_token使用CLS，训练目标只保留目标token和SEP。完全匹配评价完整答案，字符F1观察部分内容重叠；两者一起判断生成效果。",
     "最初目标中重复加入CLS会造成生成起始符号异常，改为target+SEP后重新训练。填空文件过滤缺分隔符、空输入或空目标的记录，实际数量保存到parsing字段。训练和测试均使用修正后的编码规则。",
     ["seq2seq_tasks.py qa --limit 5000 --epochs 8 --batch 8 --lr .0003 --eval-limit 500 --tag _expanded", "seq2seq_tasks.py filling --limit 5000 --epochs 8 --batch 8 --lr .0003 --eval-limit 500 --tag _expanded"]),
    ("大模型应用", ["qwen_zero_shot", "qwen_mbti"], ["plm-zero-shot", "plm-mbti"],
     "使用Qwen2.5-0.5B-Instruct进行零样本文本分类、文本匹配与条件抽取，提示提供类别schema和输出格式，不提供示例。新增评估最多100条每类留出记录，保存全部原始回答和解析结果。MBTI逐题生成A/B回答，并按上游维度映射计票。",
     "零样本任务依赖提示中的任务说明、类别含义与格式约定。类别标签通过上游label_mapping转换成可读名称。抽取使用JSON字符串数组，记录格式有效率和抽取准确率。MBTI输出按E/I、S/N、T/F、J/P分别统计，平票记问号，结果用于观察当前提示下的模型回答倾向。",
     "模型从国内镜像和ModelScope恢复，并核对文件哈希。输出保留原文，解析失败计入评价。分类、匹配和抽取分别计分，避免格式正常掩盖内容错误。",
     ["llm_tasks.py zero-shot --eval-size 100 --tag _expanded", "llm_tasks.py mbti"]),
    ("大模型训练", ["sft", "cpt", "scratch"], ["plm-sft", "plm-cpt", "plm-scratch", "plm-scratch-sft"],
     "以Qwen 0.5B完成rank-8 LoRA指令微调和继续预训练。指令数据沿用ChatGLM目录的分类与抽取混合数据，训练400条、2轮；继续预训练使用MNBVC示例文档。另用训练文档拟合3,000词Byte-BPE，从随机参数训练4层、128维GPT2，再进行ShareGPT指令微调。",
     "LoRA冻结基座权重，训练低秩增量，q_proj和v_proj使用rank8、alpha16。指令微调仅对assistant目标计算损失，prompt与padding标签为-100；继续预训练对所有非padding token预测下一个token。从零模型先学习语言建模，再用对话学习回答格式。不同tokenizer下的困惑度分别观察。",
     "4GB显存下使用BF16、LoRA、梯度累积和gradient checkpointing。长样本按保存的窗口规则截断。原6B/7B实验流程在本机以0.5B和小GPT2完成，模型规模、数据量和可训练参数写入结果配置。七类保存的生成模型均重新加载并实际预测。",
     ["llm_tasks.py sft --limit 400 --epochs 2", "llm_tasks.py cpt --limit 400 --epochs 1", "llm_tasks.py scratch --limit 1000 --epochs 5 --batch 8 --accumulate 1 --lr .0005", "llm_tasks.py scratch-sft --limit 400 --epochs 3 --batch 8 --accumulate 1"]),
]


def select_plm():
    records = results(ROOT / "transformers_tasks/local_runs/results")
    expanded = {d["name"].removesuffix("_expanded") for _, d in records if d["name"].endswith("_expanded")}
    return [(p, d) for p, d in records if d["name"] not in expanded]


def applications(records):
    text = []
    for _, row in records:
        if row.get("tasks"):
            text += ["| 零样本任务 | 指标 | 样例数 |", "|---|---|---:|"]
            for name, metrics in row["tasks"].items():
                values = {k: round(v, 4) for k, v in metrics.items() if k in ["accuracy", "exact_set_accuracy", "valid_json_rate", "f1"]}
                text.append(f"| {name} | `{json.dumps(values, ensure_ascii=False)}` | {len(metrics['examples'])} |")
            for name, metrics in row["tasks"].items():
                text += ["", "**" + name + "原始样例**", "", "```json", json.dumps(metrics["examples"][:1], ensure_ascii=False, indent=2), "```"]
            if row.get("candidates"):
                descriptions = ["原始条件prompt", "主体与关系分列", "核对属性并筛除其他关系"]
                text += ["", "### 抽取提示对照", "", f"验证集固定取{row['config']['counts']['dev']}条记录，按正例与负例分别采样。三个模板均提供任务说明与输出格式，few-shot示例数为0。模板按验证实体文字F1选择，平分时比较正负两组平均完全匹配率。", "", "| 模板 | 验证Precision | 验证Recall | 验证F1 | 正例完全匹配率 | 负例完全匹配率 |", "|---|---:|---:|---:|---:|---:|"]
                for candidate in row["candidates"]:
                    m = candidate["metrics"]
                    text.append(f"| {descriptions[candidate['template']]} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['positive_exact_set_accuracy']:.4f} | {m['negative_exact_set_accuracy']:.4f} |")
                text += ["", f"选中的模板为“{descriptions[row['selected_template']]}”。测试保持扩大实验的同100条记录，包含{row['test']['positive_samples']}条正例和{row['test']['negative_samples']}条负例。", "", "| 测试配置 | 完全匹配率 | Precision | Recall | 实体文字F1 | 负例完全匹配率 |", "|---|---:|---:|---:|---:|---:|"]
                for label, m in [("原始prompt", row["baseline_test"]), ("验证选优prompt", row["test"])]:
                    text.append(f"| {label} | {m['exact_set_accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['negative_exact_set_accuracy']:.4f} |")
                text += ["", f"固定输出空数组在这批测试记录上的完全匹配率为{row['test']['negative_samples'] / 100:.4f}，实体F1为0。结合两个指标观察正例识别与负例处理。这里按实体文字集合计分，同一文字去重，字符边界的严格匹配在UIE报告中另行评价。", "", "[完整提示模板实现](../scripts/zero_shot_prompt_study.py)"]
        if "profile" in row:
            text += ["", f"MBTI有效回答 {row['valid_responses']}/{row['total_questions']}，计票 `{json.dumps(row['votes'], ensure_ascii=False)}`，输出倾向为 `{row['profile']}`。"]
        if row.get("samples"):
            history = row["epochs"]
            text += ["", f"PPO共{len(history)}轮，保留{len(row['samples'])}条响应；首轮平均奖励{history[0]['reward']:.4f}，末轮{history[-1]['reward']:.4f}。每轮输入和采样响应会变化，结合完整奖励曲线观察更新过程。", "", "```json", json.dumps(row["samples"][:1], ensure_ascii=False, indent=2), "```"]
            if row.get("baseline_test") and row.get("test"):
                a, b = row["baseline_test"], row["test"]
                text += ["", f"固定留出提示上的{b['samples']}条生成：更新前平均奖励{a['reward_mean']:.4f}（标准差{a['reward_std']:.4f}），更新后{b['reward_mean']:.4f}（标准差{b['reward_std']:.4f}），平均变化{b['reward_mean'] - a['reward_mean']:+.4f}。这组评价使用训练外的商品和固定采样种子，得分来自同一个冻结的奖励模型。奖励模型学习了评论排序偏好，实际回答还可从商品相关性、语句通顺度和重复片段三个方面分析，更新前后的全文均保存在实验JSON中。"]
    return "\n".join(text)


def build():
    records = select_plm()
    target = REPORT / "PLM任务.md"
    main = ["# PLM任务实验报告", "", "## 实验环境与条件", "", "Python 3.11，transformers 4.57.3，PyTorch 2.6.0+cu124，RTX 3050 Ti 4GB。小BERT、UIE-base、中文T5-small、Qwen 0.5B及从零小GPT2分别承担对应实验。[环境与数据条件](实验环境.md)列出完整版本与来源。", "", "| 上游入口 | 本地实现与报告 |", "|---|---|"]
    for title, prefixes, _, *_ in GROUPS:
        main.append(f"| {title} | [{title}](PLM_{title}.md) |")
    main += ["| 工具类 | [分词与人工评价](PLM_工具.md) |", "", "## 二、实验方法与过程", "", "从BERT分类与匹配开始，保持输入、损失、训练与评价的明确对应。抽取比较预训练初始化和训练增强，Prompt比较离散模板与连续向量，T5进行实际文本生成评价。大模型部分依次执行零样本、LoRA微调、继续预训练、随机初始化语言模型及PPO。具体实现和知识点在上表的分项报告中展开。", "", "## 三、实验结果及分析", "", table(records, target), "", "按验证集选择模型后评价留出测试集。小BERT的3组UIE记录保留作初始化对照，预训练UIE的4组单独比较增强方法。", "", applications(records), "", "## 四、遇到的问题及处理情况", "", "评论和句对按内容分组处理，空测试文件与无标签测试文件采用上游dev分组留出。UIE旧模型对象转换为张量格式；T5目标移除重复起始token；生成模型重新加载保存参数进行预测。GPU训练依次执行，按4GB显存配置batch、LoRA与梯度累积。分项报告记录实际效果和处理细节。", "", "## 复现入口", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/run_plm_baselines.py", ".\\.venv\\Scripts\\python.exe scripts/quality_extra_runs.py", ".\\.venv\\Scripts\\python.exe scripts/demo_saved.py plm-bert", ".\\.venv\\Scripts\\python.exe transformers_tasks/local_runs/local_tools.py", "```", "", "各分类报告给出单项命令。代码集中在`transformers_tasks/local_runs`，上游实现与说明保留在原目录。", ""]
    target.write_text("\n".join(main), encoding="utf-8")
    overview = ROOT / "transformers_tasks/local_runs/README.md"
    overview.write_text("# PLM本地实验\n\n[完整实验报告](" + link(target, overview) + ")包含九类任务、结果、知识点与运行截图。\n\n训练脚本在本目录，结果在results，模型参数在checkpoints。环境和数据恢复步骤见根目录README。\n", encoding="utf-8")
    for title, prefixes, images, method, knowledge, issues, commands in GROUPS:
        path = REPORT / ("PLM_" + title + ".md")
        selected = [(p, d) for p, d in records if any(d["name"].startswith(prefix) for prefix in prefixes)]
        if title == "RLHF":
            ppo = next((d for _, d in selected if d["name"].startswith("qwen_ppo")), None)
            if ppo and ppo["name"].endswith("_expanded"):
                method = method.replace("12轮token级PPO更新", f"{len(ppo['epochs'])}轮token级PPO更新").replace("48条实际生成", f"{len(ppo['samples'])}条实际生成")
                commands = ["reward_task.py --limit 1000 --epochs 5", "ppo_task.py --steps 48 --heldout-eval --tag _expanded", "local_tools.py"]
        if title == "大模型应用":
            method += " 条件抽取另比较三个提示模板，验证集分别采样正负例，按实体文字F1选优，再评价扩大实验中固定的100条测试记录。"
        text = ["# " + title + "实验报告", "", "## 一、实验环境与条件", "", "依赖与设备见[实验环境](实验环境.md)。下表中的训练数量、轮数和配置来自实际运行记录，默认种子42。", "", "## 二、实验方法与过程", "", method, "", "### 实现与知识点", "", knowledge, "", "### 复现命令", "", "```powershell"]
        text += [".\\.venv\\Scripts\\python.exe transformers_tasks/local_runs/" + command for command in commands]
        text += ["```", "", "## 三、实验结果及分析", "", table(selected, path), "", applications(selected)]
        if title == "大模型应用":
            text.insert(text.index("## 三、实验结果及分析") - 1, "提示对照复现：`python scripts/zero_shot_prompt_study.py`，在扩大零样本实验完成后运行。\n")
        if title not in ["大模型应用", "RLHF"]:
            text += ["", analyses(selected, "plm", title)]
        if title in ["文本分类", "文本匹配", "Prompt", "信息抽取"]:
            curve = REPORT / ("PLM_" + title + "_curves.png")
            plot(selected, curve)
            text += ["", "### 训练曲线", "", f"![训练曲线]({curve.name})"]
        if title in ["T5生成", "大模型训练"]:
            for _, row in selected:
                curve = REPORT / (row["name"] + "_curves.png")
                plot([(p, d) for p, d in selected if d["name"] == row["name"]], curve)
                text += ["", f"![{row['name']}训练曲线]({curve.name})"]
        if title == "RLHF":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for _, row in selected:
                if not row["name"].startswith("qwen_ppo"):
                    continue
                history = row["epochs"]
                figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
                for axis, key, label in zip(axes, ["reward", "sampled_kl", "clip_fraction"], ["Mean learned reward", "Sampled token KL", "Clip fraction"]):
                    axis.plot(range(1, len(history) + 1), [epoch[key] for epoch in history], color="#315b8a")
                    axis.set_xlabel("PPO iteration")
                    axis.set_title(label)
                    axis.grid(alpha=.25)
                figure.tight_layout()
                curve = REPORT / (row["name"] + "_curves.png")
                figure.savefig(curve, dpi=150)
                plt.close(figure)
                text += ["", f"![{row['name']}更新曲线]({curve.name})"]
        if title == "信息抽取":
            text += ["", "| 初始化 / 增强 | Precision | Recall | F1 | 测试实体数 |", "|---|---:|---:|---:|---:|"]
            for _, row in selected:
                m = row["test"]
                text.append(f"| {row['name']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['gold_spans']} |")
            text += ["", "两种初始化分别成组观察。小BERT的早期swap记录采用同prompt实体替换；预训练UIE组采用同时交换主体和客体的SwapSPO。每组列出precision、recall和F1，负例挖掘主要针对误报，正例增强主要改变训练覆盖。增强前后都使用固定的原始验证与测试内容。预训练UIE的初始参数也纳入验证选优，best_epoch=0表示保留初始模型，各组仍完成记录中的全部训练轮数。", "", "[SwapSPO实际增强记录](../运行记录/quality/swap_spo_samples.json) · [Mask Then Fill实际生成与筛选记录](../运行记录/quality/mask_fill_samples.json)"]
            fill_path = ROOT / "运行记录/quality/mask_fill_samples.json"
            if fill_path.exists():
                fill = json.loads(fill_path.read_text(encoding="utf-8"))
                if fill["model"] == "t5_filling_expanded":
                    from collections import Counter
                    reasons = Counter(row.get("reason", "accepted") for row in fill["attempts"])
                    accepted = [row for row in fill["attempts"] if row["accepted"]]
                    examples = "；".join(f"“{row['removed']}”填为“{row['generated']}”" for row in accepted[:2])
                    text += ["", f"填空生成{len(fill['attempts'])}个候选，保留{fill['accepted_count']}个。其中{reasons['empty, excessive length, repeated phrase, or unchanged filling']}个因长度、空输出、重复或原样填回被筛除，{reasons['baseline extraction changed after filling']}个因抽取结果改变被筛除。保留候选的实际例子：{examples}。填空有时会补出原文未提供的信息，筛选记录与增强后的测试指标一起用于观察目标实体保持情况和上下文噪声。"]
            arms = {row["config"]["augment"]: row for _, row in selected if row["name"].startswith("uie_base_")}
            if "none" in arms and "fill" in arms:
                a, b = arms["none"]["test"], arms["fill"]["test"]
                text += ["", f"原始训练组的测试precision、recall、F1分别为{a['precision']:.4f}、{a['recall']:.4f}、{a['f1']:.4f}，填空增强组为{b['precision']:.4f}、{b['recall']:.4f}、{b['f1']:.4f}。增强后的变化需要同时结合误报和漏报观察，保留的候选数量与上下文质量也会影响这一结果。"]
        if title == "T5生成":
            lengths = ROOT / "运行记录/诊断/generation_lengths.json"
            if lengths.exists():
                data = json.loads(lengths.read_text(encoding="utf-8"))
                text += ["", "| 配置 | 测试样例 | 输入超过192 token | 目标超过64 token |", "|---|---:|---:|---:|"]
                for row in data["t5"]:
                    values = row["splits"]["test"]
                    text.append(f"| {row['name']} | {values['examples']} | {values['source_truncated']} | {values['target_truncated']} |")
                text += ["", "上表用实际tokenizer统计长度。长上下文在当前输入窗口内保留前192 token，超长训练目标保留63个内容token及SEP；完全匹配率与字符F1仍对完整原始答案计算。长文本覆盖和目标长度会影响生成结果，错误样例与这些输入条件一起分析。", "", "[实际输入长度统计](../运行记录/诊断/generation_lengths.json)"]
            text += ["", "问答与填空分开评价。字符F1高于完全匹配率时，通常表示回答包含部分正确内容，但仍有遗漏或多余字符。新增实验扩大训练与留出评估，训练轮数、批量和学习率也随配置调整。基线各评价100条，新增实验各评价500条，表中差值反映整套训练与评价配置的变化。"]
        if title == "大模型应用":
            text += ["", "分类错误结合原始评论、类别名称与模型回答逐条检查。抽取同时检查JSON格式和实体集合：可解析的回答仍可能包含其他关系的实体，两个指标分别反映格式遵循与内容识别。上面的样例直接来自本次留出评估，完整回答保存在对应JSON中。"]
            source_checks = json.loads((ROOT / "运行记录/诊断/zero_shot_output_checks.json").read_text(encoding="utf-8"))
            text += ["", "### 输出与原文核对", "", "在保存的原始回答上补做原文约束：保留文字确实出现在原文中的实体，JSON解析失败的回答维持原状态。该步骤只核对预测文字与输入，使用同一批100条记录。", "", "| 配置 / 输出处理 | 完全匹配率 | Precision | Recall | 实体文字F1 |", "|---|---:|---:|---:|---:|"]
            for record in source_checks["reports"]:
                for label, key in [("原始回答", "raw"), ("原文约束", "source_checked")]:
                    m = record[key]
                    text.append(f"| {record['model']} / {label} | {m['exact_set_accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
            removed = source_checks["reports"][0]["removed_entities"]
            text += ["", f"扩大实验中共移除{removed}个原文中没有的预测实体，实体召回率保持相同。原文中其他关系的实体仍可能被返回，因此还需结合prompt指定的主体与关系查看错误。三个提示模板的验证结果也保留在上表中，当前小模型更适合用来观察提示、输出格式和后处理的作用。", "", "[逐条输出核对](../运行记录/诊断/zero_shot_output_checks.json) · [原文约束实现](../scripts/zero_shot_output_diagnostics.py)"]
        if title == "Prompt":
            text += ["", "PET更新掩码语言模型，p-tuning仅更新8个连续prompt向量。当前少样本划分下，两者的可训练容量和标签覆盖差异会影响验证曲线。"]
        for _, row in selected:
            output_examples = row.get("examples") or row.get("test", {}).get("examples")
            if output_examples and title != "大模型应用":
                text += ["", "**" + row["name"] + "实际输出**", "", "```json", json.dumps(output_examples[:1], ensure_ascii=False, indent=2), "```"]
        for name in images:
            image = REPORT / f"运行截图/{name}.png"
            if image.exists():
                text += ["", "### 实际运行截图", "", f"![运行输出]({link(image, path)})", "", f"[原始输出]({link(image.with_suffix('.txt'), path)}) · [执行记录]({link(image.with_suffix('.json'), path)})"]
        text += ["", "## 四、遇到的问题及处理情况", "", issues, ""]
        path.write_text("\n".join(text), encoding="utf-8")
        if title == "文本分类":
            (REPORT / "PLM_分类.md").write_text("\n".join(text), encoding="utf-8")
    tool = REPORT / "PLM_工具.md"
    tool.write_text("# 分词与评价工具实验报告\n\n## 一、实验环境与条件\n\n工具基于Python本地HTTP服务，监听127.0.0.1:8765，使用中文BERT、Qwen和自训练BPE分词器。\n\n## 二、实验方法与过程\n\n输入中英文、数字和标点，查看token、编号、字符偏移与decode。用同一输入比较不同分词器，读取PPO生成样例进行人工评价。\n\n```powershell\n.\\.venv\\Scripts\\python.exe transformers_tasks/local_runs/local_tools.py\n```\n\n浏览器打开 http://127.0.0.1:8765 。\n\n## 三、实验结果及分析\n\nBERT主要按中文字符分词，Qwen和Byte-BPE按词表合并字节片段。可见token与字符数量可能不同，decode用于核对文本恢复。[接口检查](../运行记录/local_tools_checks.json)记录三个分词器及评分输入的检查结果。\n\n![分词页面](运行截图/plm-tools.png)\n\n## 四、遇到的问题及处理情况\n\n字节BPE的token字符串会呈现字节编码形式，页面同时显示decode与offset，便于观察。评分保存范围为1至5，阅读者填入后追加到human_ratings.jsonl，原始生成内容同时保存。\n", encoding="utf-8")
    print("PLM submission reports updated", len(records))
    return records


if __name__ == "__main__":
    build()
