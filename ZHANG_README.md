# Zhang 分支说明 — train.py / preprocessing.py 重建与评审修复

> 分支：`Zhang`（基于 main @ 1a751e6）
> 日期：2026-10-28
> 验证环境：Conda `astrocyte`（Python 3.12，torch 2.14.1+cpu，cellpose 4.2.1.1），纯 CPU
> 状态：全部验证通过（preprocessing 35/35，单元测试 25/25，训练自测通过，真实 Cellpose 端到端通过）

## 1. 本分支做了什么

`main` 上两个文件处于损坏状态：

- `src/train.py` 的内容被误覆盖成了一份 preprocessing 代码，**训练流水线完全丢失**；
- `src/preprocessing.py` 的 `iter_qptiff_tiles` 循环内部被误粘贴了**一整份新版文件的副本**（复制事故）。

本分支重建了这两个文件并完成其中的 TODO，随后经过两轮代码评审，修复了确认存在的问题（详见第 4 节与 `docs/experiments.md` 的 Experiment 003 / 003.v2 / 003.v3）。

## 2. 主要文件使用说明

### 2.1 `src/preprocessing.py` —— 图像加载与预处理

#### `load_image(path) -> np.ndarray`

读取**小图**（PNG/TIF/TIFF），返回 NumPy 数组。大视野 QPTIFF 不要走这里（会直接报错，引导用 `iter_qptiff_tiles`）。

| 输入 | 行为 |
| --- | --- |
| `.png` | OpenCV 读取，BGR/BGRA 自动转 RGB/RGBA；3/4 通道以外报错 |
| `.tif/.tiff` 2D 灰度 | 原样返回 `(H, W)`，dtype 保留（uint8/uint16 等） |
| `.tif/.tiff` 平面多通道 `(C,H,W)` | 按 tifffile `series.axes` 判断后转置为 `(H,W,C)`，供 `--channel` 选择 |
| `.tif/.tiff` `(1,H,W)`（Fiji 导出） | 挤压为 `(H, W)` |
| `.tif/.tiff` 含 Z/T 轴 | **报错拒绝**（防止把 z 层/时间点静默混合），需先取单层 |
| `.qptiff` / `.jpg` / 其他扩展名 | 直接 `ValueError`（扩展名校验优先于文件存在性检查） |
| 路径为空 / 文件不存在 | `ValueError` / `FileNotFoundError` |

已知限制（写在 `_load_tiff` docstring）：**无 axes 元数据的 3 层 TIFF 与平面 3 通道在形状上无法区分**（tifffile 都标 QYX），会被当作 `(H,W,3)`。导出 z-stack 时请写 `metadata={"axes": "ZYX"}`，或先取单层。

#### `prepare_image(image, channel=None) -> torch.Tensor`

把 2D 灰度或 HWC 图像转成 `[1, H, W]` 的 **float32** 张量，按 1%/99.8% 百分位归一化到 `[0, 1]`（uint16 不需要手动转 8 bit）。

- `channel=None`：1 通道直接取平面；3/4 通道按 RGB 转灰度（**这是为真 RGB 照片保留的文档化行为**）；其他通道数报错，要求显式选通道；
- `channel=N`：**任意通道数**均可（2 通道 GFAP+DAPI、5 通道等），越界/负数报错；
- 灰度图传 `channel` 会报错（配置错误，不是"取唯一通道"）；
- NaN/Inf、空数组、非数值 dtype 一律报错；
- 稀疏荧光图（前景 < 0.2% 像素）自动回退用图像最大值归一化，不会被压成全 0；真·恒定图才输出全 0。

#### `iter_qptiff_tiles(path, tile_size=512, level=0, overlap=0, channel=None, series_index=0, with_meta=False)`

基于 tifffile + zarr 的**惰性**大图切块迭代器（不整图读入内存）。

- 默认 yield `(tile, x, y)`，`tile` 为 `(tile_size, tile_size)` 的 NumPy 数组（边缘块自动裁小），`x/y` 是**所选 level** 像素坐标；
- `overlap`：重叠像素数（`0 <= overlap < tile_size`），步长 = tile_size − overlap；**重叠区的重复检测不在此处去重**，由 predict 负责；
- `channel`：多通道 slide 必须显式指定（不会默认拿第一个通道）；单通道 YX 传 channel 反而报错；
- `with_meta=True` 时 yield 字典：`tile, x, y, tile_id, row, col, level, series_index, channel, scale_xy, source`。`tile_id`（如 `r00003_c00012`）在相同 (path, series, level, tile_size, overlap, channel) 下稳定，可用于实例拼接；`scale_xy` 是所选 level 相对 level 0 的像素比例，用于坐标换算。

### 2.2 `src/train.py` —— AstrocyteUNet 训练流水线

#### 输入

1. **配置文件** `configs/config.yaml`（`--config` 可换路径）：`data.input_channels`、`model.encoder`/`output_channels`、`training.{batch_size, epochs, learning_rate, optimizer, loss, checkpoint_dir}` 等。`output_channels` 必须为 4（3 语义类 + 1 中心通道），不一致直接报错；
2. **数据批次契约**（由 `dataset.build_dataloaders()` 提供，**尚未实现，真实训练暂被阻塞**）：
   - `images`: `[B, C, H, W]` float32；
   - `targets["semantic_mask"]`: `[B, H, W]` int64，取值 0 背景 / 1 胞体 / 2 突起；
   - `targets["center_heatmap"]`: `[B, 1, H, W]` float32 ∈ [0,1]；
   - `targets["valid_mask"]`: `[B, H, W]` bool，False = 未标注/歧义像素，**不计入损失**（仓库约定不用 255 哨兵值）。

#### 损失与训练行为

- 复合损失 = 3 类语义交叉熵 + 中心点 BCE-with-logits，两项都只按 `valid_mask=True` 的像素平均；全忽略批次返回 0 而不是 NaN；
- Adam + config 学习率；按验证集 loss 早停（`early_stopping_patience`，默认 10）；指标按 batch 大小加权平均；
- `set_seed` 固定 Python/NumPy/torch 种子，有 CUDA 时同时设 cuDNN 确定性。

#### 输出

写入 `training.checkpoint_dir`（默认 `checkpoints/`）：

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 验证 loss 最优的 checkpoint |
| `last.pt` | 最后一个 epoch 的 checkpoint |
| `history.json` | 每个 epoch 的 train/val 各项指标（**每个 epoch 落盘**，中途崩溃不丢曲线） |

checkpoint 内含 `model_state_dict`、`optimizer_state_dict`、`epoch`、`metrics`、`config`、`output_heads`（输出头 schema，供 predict 拒绝旧版单热图权重）。

#### 命令行

```bash
# 合成数据自测：无需真实数据，验证训练闭环 + checkpoint 存取（默认 3 epoch，≥2）
python src/train.py --self-test --epochs 3

# 真实训练（等 dataset.build_dataloaders 落地后可用；目前会给出明确提示而非崩溃）
python src/train.py --config configs/config.yaml --device auto --epochs 50
```

两种入口都支持：`python src/train.py ...` 与 `python -m src.train ...`。

## 3. 验证结果

| 验证项 | 结果 |
| --- | ---: |
| preprocessing 功能验证（含评审回归项） | 35/35 通过 |
| `train.py --self-test`（3 epoch，CPU） | 训练损失 2.17 → 1.85，checkpoint 保存/重载正常 |
| 仓库原有单元测试（tool/） | 25/25 通过 |
| 真实 Cellpose（cpsam，CPU）256×256 / 512×512 合成荧光图 | 检出 7 / 18 个实例 |
| 用 `evaluate.py` 对照合成真值（8 px 匹配半径） | P/R/F1 = 1.0，计数 MAE = 0 |

注意：合成图上的满分只证明流水线端到端可运行，**真实精度必须用 BBBC042 / FFPE 真实图像另行评估**。测试图为单通道 uint16 TIFF 原图 + 256/512 裁剪图（按团队约定，未用 JPG/截图）。

## 4. 其余文件的问题与修改

### `src/baseline_cellpose.py`

- **问题**：多通道图像不给 `--channel` 时，`prepare_image` 会把荧光通道按 RGB 公式（0.299R+0.587G+0.114B）静默混成灰度。"荧光通道必须显式选择"这条规则的强制点应在调用方，而不是 preprocessing（preprocessing 需要对真 RGB 照片保留灰度化行为）。
- **修改**：CLI 入口增加守卫——加载后若图像为多通道且未给 `--channel`，直接退出并提示"fluorescence channels must not be mixed as RGB grayscale"。实测：3 通道 TIFF 不给 channel 报错退出；给 `--channel 0` 正常检出 7 个实例。

### `tool/check_dataset.py`

- **问题**：`IMAGE_EXTENSIONS` 含 `.jpg/.jpeg`，与"只支持 png/tif/tiff"的约定冲突——jpg 能通过数据集清单检查，到 `load_image` 才报错，问题发现得太晚。
- **修改**：扩展名收紧为 `{.tif, .tiff, .png}`；新增 `unsupported_files_ignored` 报告项，混入的 jpg 等文件会被逐一点名列出（忽略但不静默）。

### `configs/config.yaml`

- **问题**：`model.output_channels: 1` 是旧单热图设计的残留值，与 REDESIGN 后模型的实际输出（3 语义 + 1 中心 = 4 通道）不符，且没有任何校验。
- **修改**：更新为 `output_channels: 4` 并注明含义；`train.build_model` 增加校验，配置与模型契约不一致时直接报错。

### `docs/experiments.md`

- 追加三条实验记录，完整覆盖本分支工作：
  - **Experiment 003**：两个文件重建 + 端到端验证（27 项预处理检查、训练自测、25 个单测、真实 Cellpose、合成真值评估）；
  - **Experiment 003.v2**：第一轮评审修复（A1/A2/A4/A5/B1/B3/B5/B6/B7 + 2 项附加）与 35 项复验；
  - **Experiment 003.v3**：第二轮评审修复（check_dataset 扩展名、baseline 守卫、QYX 歧义文档化）。

## 5. 遗留工作（不在本分支范围）

- `src/dataset.py`：`AstrocyteDataset`、标注解析、热图生成、`build_dataloaders` 全部为桩 —— **真实训练的唯一阻塞项**；
- `src/postprocessing.py`：`extract_centers` 峰值提取未实现；
- `src/predict.py`：推理全链路为桩；
- `tool/check_dataset.py` 的 `validate_coordinates` 待确认 BBBC042 positions 格式后实现；
- 评审中判定为"约定/范围差异"而未改的条目（A3 RGB 灰度化文档行为、B2 valid_mask 约定、B4 torch≥2.0、A6 tile_id 作用域），及两条低优先级遗留（A6 加固、B4 兜底），见 Experiment 003.v2 的处置表。
