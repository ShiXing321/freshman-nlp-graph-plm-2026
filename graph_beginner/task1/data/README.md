# 节点分类数据

Cora、CiteSeer和Flickr的原始数据存放在本任务的`raw`目录，PyG处理结果由首次加载生成。

在项目根目录恢复数据：

```powershell
.\.venv\Scripts\python.exe graph_beginner/prepare_data.py Cora
.\.venv\Scripts\python.exe graph_beginner/prepare_data.py CiteSeer
.\.venv\Scripts\python.exe graph_beginner/prepare_data.py Flickr
```

Cora与CiteSeer使用Planetoid公开节点mask；Flickr保留官方role.json。下载来源、校验和划分在实验报告中记录。
