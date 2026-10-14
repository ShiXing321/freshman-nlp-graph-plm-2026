# 链路预测数据

本任务与节点分类共用`graph_beginner/task1/data/raw`中的Cora、CiteSeer和Flickr原始图。恢复命令见[节点数据](../../task1/data/README.md)。

训练脚本通过RandomLinkSplit重新划分边，验证和测试各占10%，训练正边的20%作为独立监督边。节点分类的标签mask用于任务一，链路预测使用这里的边划分。
