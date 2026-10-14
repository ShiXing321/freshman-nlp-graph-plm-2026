# 知识图谱数据

使用FB15k-237官方train、valid和test划分。`code/train.py`首次运行会通过GitHub公共API取得原始三元组，保存到`raw/FB15k-237`。

```powershell
.\.venv\Scripts\python.exe graph_beginner/task4/code/train.py --model transe --epochs 20 --limit 0 --eval-limit 200 --test-limit 0 --batch 1024 --negatives 8 --tag _full
```

训练只使用train三元组；filtered测试采用全部已知真三元组过滤其他正确答案。结果配置分别记录训练数量、验证采样数量和实际测试数量。
