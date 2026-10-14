# 图级预测数据

MUTAG使用PyG TUDataset，ZINC使用官方full划分的国内镜像。数据存放在`raw/MUTAG`和`raw/ZINC-HF`。

```powershell
.\.venv\Scripts\python.exe graph_beginner/prepare_data.py MUTAG
.\.venv\Scripts\python.exe graph_beginner/prepare_data.py ZINC
```

MUTAG按图划分80/10/10；ZINC分别从train、validation、test固定采样2,000、1,000、1,000张图。ZINC目标为连续性质，以MAE评价。
