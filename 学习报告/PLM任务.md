# PLM任务实验报告

## 实验环境与条件

Python 3.11，transformers 4.57.3，PyTorch 2.6.0+cu124，RTX 3050 Ti 4GB。小BERT、UIE-base、中文T5-small、Qwen 0.5B及从零小GPT2分别承担对应实验。[环境与数据条件](实验环境.md)列出完整版本与来源。

| 上游入口 | 本地实现与报告 |
|---|---|
| 文本分类 | [文本分类](PLM_文本分类.md) |
| 文本匹配 | [文本匹配](PLM_文本匹配.md) |
| 信息抽取 | [信息抽取](PLM_信息抽取.md) |
| Prompt | [Prompt](PLM_Prompt.md) |
| RLHF | [RLHF](PLM_RLHF.md) |
| T5生成 | [T5生成](PLM_T5生成.md) |
| 大模型应用 | [大模型应用](PLM_大模型应用.md) |
| 大模型训练 | [大模型训练](PLM_大模型训练.md) |
| 工具类 | [分词与人工评价](PLM_工具.md) |

## 二、实验方法与过程

从BERT分类与匹配开始，保持输入、损失、训练与评价的明确对应。抽取比较预训练初始化和训练增强，Prompt比较离散模板与连续向量，T5进行实际文本生成评价。大模型部分依次执行零样本、LoRA微调、继续预训练、随机初始化语言模型及PPO。具体实现和知识点在上表的分项报告中展开。

## 三、实验结果及分析

| 实验配置 | 训练规模 | 轮数 | 验证指标 | 测试指标 | 运行时间 / s |
|---|---:|---:|---|---|---:|
| [classification_n0_lr0.0003](../transformers_tasks/local_runs/results/classification_n0_lr0.0003.json) | 402 | 20 | macro_f1=0.7175 | accuracy=0.7188; macro_f1=0.7780 | 7.0 |
| [cpt](../transformers_tasks/local_runs/results/cpt.json) | 200 | 1 | loss=3.2844 | loss=3.1278; perplexity=22.8239 | 55.4 |
| [dssm_n0_lr0.0003_grouped](../transformers_tasks/local_runs/results/dssm_n0_lr0.0003_grouped.json) | 923 | 15 | macro_f1=0.8768 | accuracy=0.8229; macro_f1=0.7935; auc=0.9035; average_precision=0.8456 | 21.0 |
| [pet_n0_lr0.0003](../transformers_tasks/local_runs/results/pet_n0_lr0.0003.json) | 63 | 30 | macro_f1=0.4531 | accuracy=0.6169; macro_f1=0.4561 | 8.6 |
| [pointwise_n0_lr0.0003_grouped](../transformers_tasks/local_runs/results/pointwise_n0_lr0.0003_grouped.json) | 923 | 15 | macro_f1=0.8305 | accuracy=0.8286; macro_f1=0.7948; auc=0.9040; average_precision=0.8212 | 11.9 |
| [ptuning_n0_lr0.01](../transformers_tasks/local_runs/results/ptuning_n0_lr0.01.json) | 63 | 30 | macro_f1=0.2197 | accuracy=0.3684; macro_f1=0.2389 | 6.9 |
| [qwen_mbti](../transformers_tasks/local_runs/results/qwen_mbti.json) | — | — | — | — | 7.6 |
| [qwen_ppo_v2_expanded](../transformers_tasks/local_runs/results/qwen_ppo_v2_expanded.json) | 192 | 48 | — | — | 470.7 |
| [qwen_zero_shot_prompt_study](../transformers_tasks/local_runs/results/qwen_zero_shot_prompt_study.json) | 0 | — | f1=0.2276 | f1=0.0788 | 166.9 |
| [qwen_zero_shot_v2_expanded](../transformers_tasks/local_runs/results/qwen_zero_shot_v2_expanded.json) | — | — | — | — | 101.9 |
| [reward_n1000](../transformers_tasks/local_runs/results/reward_n1000.json) | 6000 | 5 | preference_accuracy=0.8350 | preference_accuracy=0.8425 | 41.3 |
| [sbert_n0_lr0.0003_grouped](../transformers_tasks/local_runs/results/sbert_n0_lr0.0003_grouped.json) | 923 | 15 | macro_f1=0.8861 | accuracy=0.9143; macro_f1=0.8956; auc=0.9713; average_precision=0.9328 | 21.3 |
| [scratch-sft](../transformers_tasks/local_runs/results/scratch-sft.json) | 400 | 3 | loss=6.0896 | loss=5.9943; perplexity=401.1337 | 4.3 |
| [scratch](../transformers_tasks/local_runs/results/scratch.json) | 200 | 5 | loss=6.6654 | loss=6.7755; perplexity=876.1420 | 3.0 |
| [sft](../transformers_tasks/local_runs/results/sft.json) | 400 | 2 | loss=0.3106 | loss=0.2817; perplexity=1.3254 | 202.6 |
| [simcse_n10000_lr0.0003](../transformers_tasks/local_runs/results/simcse_n10000_lr0.0003.json) | 10000 | 5 | macro_f1=0.6171 | accuracy=0.6150; macro_f1=0.5949; auc=0.6465; average_precision=0.6109 | 58.8 |
| [t5_filling_n1000_v2](../transformers_tasks/local_runs/results/t5_filling_n1000_v2.json) | 1000 | 10 | loss=4.0089 | loss=4.1071; exact_match=0.0300; character_f1=0.2577 | 179.0 |
| [t5_filling_n5000_v2_expanded](../transformers_tasks/local_runs/results/t5_filling_n5000_v2_expanded.json) | 5000 | 8 | loss=3.1771 | loss=3.1253; exact_match=0.0360; character_f1=0.2913 | 692.2 |
| [t5_qa_n1000_v2](../transformers_tasks/local_runs/results/t5_qa_n1000_v2.json) | 1000 | 10 | loss=2.3960 | loss=2.7285; exact_match=0.2700; character_f1=0.5462 | 212.0 |
| [t5_qa_n5000_v2_expanded](../transformers_tasks/local_runs/results/t5_qa_n5000_v2_expanded.json) | 5000 | 8 | loss=1.4798 | loss=1.3002; exact_match=0.3720; character_f1=0.6503 | 852.8 |
| [uie_base_fill_v1](../transformers_tasks/local_runs/results/uie_base_fill_v1.json) | 2174 | 8 | f1=0.8000 | f1=0.8696 | 522.3 |
| [uie_base_negative_v1](../transformers_tasks/local_runs/results/uie_base_negative_v1.json) | 2102 | 8 | f1=0.7458 | f1=0.9091 | 511.9 |
| [uie_base_none_v1](../transformers_tasks/local_runs/results/uie_base_none_v1.json) | 2087 | 8 | f1=0.7500 | f1=0.9091 | 502.2 |
| [uie_base_swap_v1](../transformers_tasks/local_runs/results/uie_base_swap_v1.json) | 2227 | 8 | f1=0.7541 | f1=0.8372 | 540.4 |
| [uie_negative_v3](../transformers_tasks/local_runs/results/uie_negative_v3.json) | 2194 | 15 | f1=0.1250 | f1=0.0000 | 54.9 |
| [uie_none_v3](../transformers_tasks/local_runs/results/uie_none_v3.json) | 2087 | 15 | f1=0.2000 | f1=0.3448 | 52.5 |
| [uie_swap_v3](../transformers_tasks/local_runs/results/uie_swap_v3.json) | 2427 | 15 | f1=0.1176 | f1=0.1463 | 65.5 |

按验证集选择模型后评价留出测试集。小BERT的3组UIE记录保留作初始化对照，预训练UIE的4组单独比较增强方法。


MBTI有效回答 93/93，计票 `{"J": 15, "S": 13, "E": 13, "F": 10, "I": 8, "P": 7, "N": 14, "T": 13}`，输出倾向为 `ENTJ`。

PPO共48轮，保留192条响应；首轮平均奖励4.7941，末轮3.1243。每轮输入和采样响应会变化，结合完整奖励曲线观察更新过程。

```json
[
  {
    "iteration": 1,
    "prompt": "写一条正面的购物评论，只写评论。商品：苹果。",
    "response": "我非常喜欢这款苹果手机！设计精美，性能强大，启动速度快，并且能高效地处理各种任务和游戏。我认为它是一款值得购买的好手机",
    "reward": 4.9072675704956055,
    "sampled_kl": 0.0,
    "tokens": 32
  }
]
```

固定留出提示上的16条生成：更新前平均奖励3.8594（标准差0.7707），更新后3.3643（标准差1.5263），平均变化-0.4951。这组评价使用训练外的商品和固定采样种子，得分来自同一个冻结的奖励模型。奖励模型学习了评论排序偏好，实际回答还可从商品相关性、语句通顺度和重复片段三个方面分析，更新前后的全文均保存在实验JSON中。
| 零样本任务 | 指标 | 样例数 |
|---|---|---:|
| extraction | `{"exact_set_accuracy": 0.02, "valid_json_rate": 1.0, "f1": 0.0788}` | 100 |

**extraction原始样例**

```json
[
  {
    "prompt": "李泳豪的出生地",
    "content": "李泳豪是艺人李家鼎和施明的儿子",
    "gold": [],
    "output": "```json\n[\n    \"李家鼎\",\n    \"施明\"\n]\n```",
    "prediction": [
      "李家鼎",
      "施明"
    ],
    "correct": false
  }
]
```

### 抽取提示对照

验证集固定取40条记录，按正例与负例分别采样。三个模板均提供任务说明与输出格式，few-shot示例数为0。模板按验证实体文字F1选择，平分时比较正负两组平均完全匹配率。

| 模板 | 验证Precision | 验证Recall | 验证F1 | 正例完全匹配率 | 负例完全匹配率 |
|---|---:|---:|---:|---:|---:|
| 原始条件prompt | 0.1373 | 0.6667 | 0.2276 | 0.2000 | 0.0000 |
| 主体与关系分列 | 0.0870 | 0.0952 | 0.0909 | 0.0500 | 0.0000 |
| 核对属性并筛除其他关系 | 0.1122 | 0.5238 | 0.1849 | 0.0500 | 0.0000 |

选中的模板为“原始条件prompt”。测试保持扩大实验的同100条记录，包含15条正例和85条负例。

| 测试配置 | 完全匹配率 | Precision | Recall | 实体文字F1 | 负例完全匹配率 |
|---|---:|---:|---:|---:|---:|
| 原始prompt | 0.0200 | 0.0430 | 0.4706 | 0.0788 | 0.0000 |
| 验证选优prompt | 0.0200 | 0.0430 | 0.4706 | 0.0788 | 0.0000 |

固定输出空数组在这批测试记录上的完全匹配率为0.8500，实体F1为0。结合两个指标观察正例识别与负例处理。这里按实体文字集合计分，同一文字去重，字符边界的严格匹配在UIE报告中另行评价。

[完整提示模板实现](../scripts/zero_shot_prompt_study.py)
| 零样本任务 | 指标 | 样例数 |
|---|---|---:|
| classification | `{"accuracy": 0.4062}` | 32 |
| matching | `{"accuracy": 0.69}` | 100 |
| extraction | `{"exact_set_accuracy": 0.02, "valid_json_rate": 1.0}` | 100 |

**classification原始样例**

```json
[
  {
    "input": "手感不错，用料也很好，不知道水洗后怎样，相信大品牌，质量过关，五星好评！！！",
    "gold": "衣服",
    "output": "电脑",
    "correct": false
  }
]
```

**matching原始样例**

```json
[
  {
    "a": "我的签名怎么写才好看",
    "b": "英语的字体怎么写的好看点。",
    "gold": 0,
    "output": "不同。",
    "prediction": 0,
    "correct": true
  }
]
```

**extraction原始样例**

```json
[
  {
    "prompt": "李泳豪的出生地",
    "content": "李泳豪是艺人李家鼎和施明的儿子",
    "gold": [],
    "output": "```json\n[\n    \"李家鼎\",\n    \"施明\"\n]\n```",
    "prediction": [
      "李家鼎",
      "施明"
    ],
    "correct": false
  }
]
```

## 四、遇到的问题及处理情况

评论和句对按内容分组处理，空测试文件与无标签测试文件采用上游dev分组留出。UIE旧模型对象转换为张量格式；T5目标移除重复起始token；生成模型重新加载保存参数进行预测。GPU训练依次执行，按4GB显存配置batch、LoRA与梯度累积。分项报告记录实际效果和处理细节。

## 复现入口

```powershell
.\.venv\Scripts\python.exe scripts/run_plm_baselines.py
.\.venv\Scripts\python.exe scripts/quality_extra_runs.py
.\.venv\Scripts\python.exe scripts/demo_saved.py plm-bert
.\.venv\Scripts\python.exe transformers_tasks/local_runs/local_tools.py
```

各分类报告给出单项命令。代码集中在`transformers_tasks/local_runs`，上游实现与说明保留在原目录。
