# NLP、Graph与PLM新生任务

三个方向的实验代码、报告、结果与运行记录按任务放在对应目录。

| 方向 | 内容 | 报告入口 |
|---|---|---|
| NLP | NumPy分类、PyTorch分类、ESIM、BiLSTM-CRF、字符语言模型 | [五项任务报告](学习报告/总报告.md) |
| Graph | 节点分类、链路预测、图级预测、知识图谱补全 | [节点](graph_beginner/task1/README.md) · [链路](graph_beginner/task2/README.md) · [图级](graph_beginner/task3/README.md) · [知识图谱](graph_beginner/task4/README.md) |
| PLM | 匹配、抽取、Prompt、分类、RLHF、生成、大模型应用与训练、工具 | [实验报告](学习报告/PLM任务.md) |

[总报告](学习报告/总报告.md) · [实验环境](学习报告/实验环境.md) · [原始运行截图](学习报告/运行截图) · [逐项日志](进度日志.md) · [工具上手](学习报告/工具上手.md)

## 目录

```text
nlp-beginner/legacy/        NLP五项实现、报告与结果
graph_beginner/task*/      每项的code、README与results
transformers_tasks/local_runs/  PLM本地实验
common/                    随机种子、下载与结果记录
scripts/                   数据恢复、实验队列、核查与截图
学习报告/                   环境、实验分析与知识点
运行记录/                   原始日志及检查结果
```

## 运行

先安装Python 3.11，再创建环境：

```powershell
.\setup.ps1
```

默认新建`.venv-reproduce`。下面以已经配置好的`.venv`为例，在项目根目录执行；新环境将路径中的`.venv`换成`.venv-reproduce`。

```powershell
.\.venv\Scripts\python.exe scripts/prepare_submission_data.py nlp
.\.venv\Scripts\python.exe scripts/prepare_submission_data.py graph
.\.venv\Scripts\python.exe scripts/prepare_submission_data.py plm
.\.venv\Scripts\python.exe scripts/prepare_submission_data.py models
.\.venv\Scripts\python.exe scripts/prepare_submission_data.py uie
.\.venv\Scripts\python.exe scripts/quality_runs.py
.\.venv\Scripts\python.exe scripts/run_plm_baselines.py
.\.venv\Scripts\python.exe scripts/quality_extra_runs.py
.\.venv\Scripts\python.exe scripts/quality_followup_runs.py
.\.venv\Scripts\python.exe scripts/zero_shot_prompt_study.py
```

训练按队列依次执行，成功命令保存在history.jsonl；再次运行时同时检查结果与模型参数，文件完整的实验会跳过。单项训练命令在各报告第二节。神经网络训练使用CUDA，NumPy分类使用CPU；下载优先采用国内镜像。

Graph基础对照的复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/run_matrix.py graph-tu
.\.venv\Scripts\python.exe scripts/run_matrix.py graph-zinc
```

下载Flickr和ZINC时，如默认站点连接失败，使用各任务报告与数据恢复脚本中记录的镜像。

检查算法、读取模型和查看分词：

```powershell
.\.venv\Scripts\python.exe scripts/verify_algorithms.py
.\.venv\Scripts\python.exe scripts/verify_plm_math.py
.\.venv\Scripts\python.exe scripts/verify_kge_ranking.py
.\.venv\Scripts\python.exe scripts/demo_saved.py nlp1
.\.venv\Scripts\python.exe transformers_tasks/local_runs/local_tools.py
```

工具页面地址为 http://127.0.0.1:8765 。报告图片由实际执行页面截取，附带对应原始输出与执行记录。GitHub保存代码、报告、截图和结构化结果。数据集与预训练权重由下载脚本恢复，训练检查点由实验命令生成。

## 题目与来源

- NLP：[FudanNLP/nlp-beginner](https://github.com/FudanNLP/nlp-beginner)，采用legacy原版五项任务。
- Graph：[Maxioo/graph_beginner](https://github.com/Maxioo/graph_beginner)。
- PLM：[HarderThenHarder/transformers_tasks](https://github.com/HarderThenHarder/transformers_tasks)。

上游仓库与预训练模型的固定版本见[来源版本](来源版本.json)。本机实验根据4GB显存配置模型与数据规模，具体条件、效果和偏差在对应报告中说明。
