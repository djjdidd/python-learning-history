# 新增实验与日常提交

[返回首页](../README.md)

## 文件放置与命名

- 科学计算库练习放到 `01_scientific_python/`。
- 本科和暑期记录分别放到 `02_undergraduate/`、`03_summer_review/`。
- 第一阶段光刻实验放到 `04_lithography_stage1/`；后续进入新阶段时再建对应目录。
- 临时但希望保留的独立实验放到 `05_experiments/`，文件名直接说明用途。

新文件可用 `day25_topic_name.py`，统一小写字母、下划线和两位 day 编号。同一天有不同版本时，加上能够区分模型或实验的后缀。添加后，更新所属目录的 README 索引；首页脚本数量是本次整理时的快照。

## 每个实验记录什么

在脚本开头的注释中写下：实验问题、输入参数、输出图/指标、主要公式和预期现象。运行后记录关键参数、实际结果与尚未理解的地方。先改变一个参数，便于解释原因。

可再生成的输出放入 `results/`，默认被 Git 忽略。确实需要保留的代表性结果可另建 `docs/figures/`，连同参数和结论一起提交。

## 在 VS Code 终端提交

先确认当前分支和改动：

```bash
git status
git diff
```

确认后添加本次相关文件，例如：

```bash
git add "04_lithography_stage1/day25_topic_name.py" "04_lithography_stage1/README.md"
git commit -m "Add day25 experiment and learning notes"
git push
```

上面的 day25 路径是命名示例，使用时替换为实际存在的文件。

## 获取本次整理

本次修改在 `organize/learning-history-20261013` 分支，可通过 Pull Request 查看全部改动。整理合并到 main 后，在本机先查看 `git status`，提交或暂存未完成修改，再执行：

```bash
git switch main
git pull --ff-only origin main
```

合并前也可以获取整理分支查看：

```bash
git fetch origin
git switch --track origin/organize/learning-history-20261013
```

若本地已存在该分支，直接 `git switch organize/learning-history-20261013`。
