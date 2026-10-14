"""Append the completed submission inventory and verification outcomes."""
import json
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    for name in ["zero-shot-study.log", "zero-shot-study.err"]:
        original = ROOT / name
        if original.exists():
            folder = ROOT / "运行记录/quality-zero-shot"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(original.read_bytes())
    inventory = json.loads((ROOT / "学习报告/正式实验索引.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "运行记录/提交核查.json").read_text(encoding="utf-8"))
    if audit["status"] != "passed":
        raise RuntimeError("Complete the submission checks before writing the final log")
    counts = {key: sum(row["file"].startswith(folder) for row in inventory) for key, folder in
        [("NLP", "nlp-beginner/"), ("Graph", "graph_beginner/"), ("PLM", "transformers_tasks/")]}
    text = ["", "## " + datetime.now().astimezone().isoformat() + "：实验、报告与提交核查完成", "",
        f"正式实验索引收录{len(inventory)}组记录：NLP {counts['NLP']}组、Graph {counts['Graph']}组、PLM {counts['PLM']}组。NLP五项、Graph四项和PLM九类入口均有实现与实验报告。保留22张实际执行页面截屏，配套原始输出、执行命令与图片哈希。", "",
        "Kaggle使用原156,060条短语并按SentenceId分组。SNLI使用10万条训练和完整验证、测试划分；CoNLL使用完整句子，补充大小写特征配置。图实验包含模型、采样、池化、学习率和层数对照，Cora GCN补充三个种子。知识图谱使用全部训练三元组与完整测试排名。", "",
        "PLM保留基础实验，补充原UIE预训练权重与四组增强、5,000条训练的T5问答与填空、扩大零样本评价以及48轮PPO的固定留出提示前后比较。条件抽取增加三个prompt的验证集对照，并保留相同100条测试记录比较。模型规模和数据条件在分项报告中记录。", "",
        f"提交核查的{len(audit['checks'])}项检查通过。核查涵盖任务覆盖、完整留出评价、数学计算、模型重新加载、数据模型来源、独立环境、报告章节、链接与运行截图。结果见[提交核查](运行记录/提交核查.json)，每项运行日志按队列保存在运行记录目录。", "",
        "[正式实验索引](学习报告/正式实验索引.json) · [总报告](学习报告/总报告.md) · [工具练习](学习报告/工具上手.md)", ""]
    with (ROOT / "进度日志.md").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(text))
    print(json.dumps({"experiments": len(inventory), "counts": counts, "checks": len(audit["checks"])}))


if __name__ == "__main__":
    main()
