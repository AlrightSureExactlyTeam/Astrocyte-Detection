# 真实 GFAP ROI 评估流水线

本流水线用于在原始 GFAP ROI 图像到位后，比较已训练 U-Net 与 Cellpose 的**细胞中心定位**。当前 BBBC042 只有点标注，U-Net 的语义头没有有效监督，因此真实 ROI 流程只使用中心热图，不用语义输出过滤检测结果。

## 1. 放置数据

```text
datasets/real_slide_test/
├── images/          # 原始、无手绘标记的 PNG/TIF/TIFF
├── annotations/     # 与图像同 stem 的 CSV
├── overlays/        # 可选的人工标注参考，不参与计算
└── metadata/        # QPTIFF 来源、ROI 位置、像素间距、通道和坐标变换记录
```

例如 `images/roi_01.tif` 对应 `annotations/roi_01.csv`：

```csv
cell_id,x,y,status
1,677,291,confirmed
2,134,450,confirmed
6,394,66,uncertain
```

坐标必须是原始 ROI 的 `(x,y)` 像素坐标。不要根据文件名猜测 QPTIFF 坐标关系；请在 metadata 中保留真实来源与变换记录。多通道荧光图需要在配置中显式指定 GFAP 通道。

## 2. 常用命令

以下命令均在项目根目录执行，默认只使用 CPU。

```powershell
# 数据校验：图像未到位时也生成报告，不伪造精度
python tool/run_experiment.py --stage validate --experiment-dir results/experiments/data_check

# U-Net 预测；换成实际 checkpoint 路径
python tool/run_experiment.py --stage predict --model unet --checkpoint checkpoints/RUN_NAME/best.pt

# Cellpose 预测
python tool/run_experiment.py --stage predict --model cellpose

# 在已有实验目录中补充另一模型、评估、比较、报告
python tool/run_experiment.py --stage predict --model cellpose --experiment-dir results/experiments/EXPERIMENT_NAME
python tool/run_experiment.py --stage evaluate --experiment-dir results/experiments/EXPERIMENT_NAME --match-distance 8
python tool/run_experiment.py --stage compare --experiment-dir results/experiments/EXPERIMENT_NAME
python tool/run_experiment.py --stage report --experiment-dir results/experiments/EXPERIMENT_NAME

# 完整评估：校验 + 两个模型预测 + 评估 + 比较 + 图表 + 报告；不会训练
python tool/run_experiment.py --stage all --checkpoint checkpoints/RUN_NAME/best.pt --match-distance 8

# 显式单独训练（仅在你确认需要时执行）
python tool/run_experiment.py --stage train
```

也可以使用独立入口：`tool/validate_real_data.py`、`tool/predict_real_images.py`、`tool/run_cellpose_baseline.py`、`tool/evaluate_predictions.py`、`tool/compare_models.py`、`tool/visualize_comparison.py` 和 `tool/generate_report.py`。

## 3. 输出与防覆盖

省略 `--experiment-dir` 的预测或完整评估会创建唯一目录：

```text
results/experiments/pilot_YYYYMMDD_HHMMSS/
├── validation/validation_report.json
├── unet/{predictions,overlays,metadata}/
├── cellpose/{predictions,overlays,metadata}/
├── metrics/{unet_metrics.csv,cellpose_metrics.csv,metrics.csv,comparison.csv,summary.json}
├── figures/
├── report.md
└── experiment_config.yaml
```

未来每次训练也会写入 `checkpoints/run_YYYYMMDD_HHMMSS/`，保存 `best.pt`、`last.pt`、`history.json` 和 `config_snapshot.yaml`，不会覆盖已有运行。

## 4. 评估规则

- 先将预测与 confirmed 标注进行一对一匹配，再处理 unmatched prediction 与 uncertain 标注。
- 与 uncertain 匹配的剩余预测记为 ignored，不计入主要 FP；uncertain 不会变成 TP，也不会压制 confirmed 匹配。
- 匹配距离通过 `--match-distance` 配置并写入结果；应结合像素间距和人工标注误差选择。
- Micro-F1 由所有 ROI 汇总后的 TP/FP/FN 计算，不是各 ROI F1 的平均值。
- 只运行一个模型时会保留其结果，并明确标记比较不完整。
- 计数误差为 `有效预测数（排除 ignored）− confirmed 标注数`，只有在人工 confirmed 标注覆盖足够完整时才具有计数性能意义。

## 5. 科学解释限制

这是同一组织切片多个 ROI 的 pilot evaluation，不能据此声称跨切片泛化或将 ROI 当作独立生物学重复。Cellpose 是通用细胞分割基线，其对象不一定都是 GFAP 阳性星形胶质细胞。没有语义 mask 时，不报告语义分割性能；合成数据测试只证明代码路径可运行，不代表真实检测精度。

## 6. 测试

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s tool -p "test_*.py"
```

真实 QPTIFF 不参与自动化单元测试。真实 checkpoint 缺失或 schema 不兼容时，预测会明确失败，不会回退到随机权重。
