# Zhang 分支说明 — 全模块实现：preprocessing / train / dataset / postprocessing / predict

> 分支：`Zhang`（基于 main @ 1a751e6）
> 日期：2026-10-28
> 验证环境：Conda `astrocyte`（Python 3.12，torch 2.14.1+cpu，cellpose 4.2.1.1），纯 CPU
> 状态：全部验证通过（preprocessing 35/35，单元测试 25/25，训练自测通过，真实 Cellpose 端到端通过；dataset/postprocessing/predict 端到端及评审回归 37/37 通过）

## 1. 本分支做了什么

`main` 上两个文件处于损坏状态：

- `src/train.py` 的内容被误覆盖成了一份 preprocessing 代码，**训练流水线完全丢失**；
- `src/preprocessing.py` 的 `iter_qptiff_tiles` 循环内部被误粘贴了**一整份新版文件的副本**（复制事故）。

本分支重建了这两个文件并完成其中的 TODO，随后经过两轮代码评审，修复了确认存在的问题（详见第 4 节与 `docs/experiments.md` 的 Experiment 003 / 003.v2 / 003.v3）。

**第二阶段（本次提交）**：按项目 REDESIGN 契约实现了剩余三个桩文件的全部 TODO——`src/dataset.py`（25 个 TODO）、`src/postprocessing.py`（6 个 TODO）、`src/predict.py`（26 个 TODO）。至此**真实数据训练与自家模型推理两条链路均已打通**（Experiment 004）。

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
python src/train.py --config configs/config.yaml --device cuda --epochs 50
```

两种入口都支持：`python src/train.py ...` 与 `python -m src.train ...`。训练默认读取 `training.device: cuda`，要求 NVIDIA GPU、兼容驱动和 CUDA 版 PyTorch；省略 `--device` 也会使用 CUDA，不会再静默回退到 CPU。可用 `--device cuda:1` 选择指定显卡。

真实训练现在已可直接运行（`build_dataloaders` 已实现），无需再等 stub。

### 2.3 `src/dataset.py` —— 数据集与 DataLoader

#### 输入

- `images_dir`：`datasets/BBBC042/images`（png/tif/tiff，按文件名 stem 配对）；
- `annotations_dir`：`datasets/BBBC042/positions`（每个图像对应一个 `<stem>.txt`，每行一个中心点，取前两列数值为 `(x, y)`，支持逗号/制表/空格分隔，`#`/`//` 注释与表头自动跳过；**空文件 = 确认无细胞**，**缺文件 = 报错**，绝不把缺标注当无细胞）；
- 可选 `masks_dir`：若存在 `<stem>_semantic.tif` + `<stem>_instance.tif` 专家掩码（必须成对），该样本转为全监督。

#### 点标注监督策略（REDESIGN 核心）

BBBC042 只有点标注，无法监督胞体/突起分割。点标注样本采用**部分监督**：

- 点半径 `soma_radius_pixels`（默认 5 px）内的圆盘 → `semantic_mask=1`（胞体），valid；
- 距所有点 > `background_distance_pixels`（默认 16 px）→ `semantic_mask=0`（可信背景），valid；
- 两者之间的环带（很可能是突起/交叉歧义区）→ `valid_mask=False`，**不计入损失**；
- **绝不从点标注合成突起（2）标签**；`center_heatmap` 照常由全部点生成（`heatmap.sigma_pixels`，默认 3.0，与训练约定一致）。

#### 输出（每个样本）

`(image, targets, metadata)`：

- `image`：`[C,H,W]` float32（经 `prepare_image` 归一化）；
- `targets`：`semantic_mask [H,W] int64`、`instance_mask [H,W] int64`、`center_heatmap [1,H,W] float32`、`valid_mask [H,W] bool`（点标注样本 `instance_mask` 全零，掩码样本含实例 ID；当前复合损失不消费实例掩码，但实例标注会完整保留在批次中）；
- `metadata`：`sample_id / annotation_kind(points|masks|empty) / tile_origin / level / scale / n_points / n_points_out_of_bounds`。

几何一致性：训练随机裁剪 + 翻转/90° 旋转（确定性逐样本种子，跨运行可复现）；**验证/测试使用确定性全覆盖网格裁剪**（包含右/下边缘，不再只验证中心区域）；**图像、三类掩码、点坐标用同一套几何变换**，实例 ID 不变；裁剪后越界的点被丢弃；小图先补零、补零区 valid=False。若一张图的标注点**全部越界**，直接报错提示检查坐标约定，不会静默降级成"确认无细胞"。划分按**源图像**进行（`train_fraction/val_fraction`，种子固定），防止泄漏。

#### `build_dataloaders(config) -> (train_loader, val_loader)`

从 config.yaml 读取全部参数，返回可直接喂给 `train.run_epoch()` 的 DataLoader（含自定义 collate，metadata 保持字典形式）。

### 2.4 `src/postprocessing.py` —— 峰值提取与实例分离

- `extract_centers(probability_map, threshold=0.5, min_distance=5) -> (N,2) (x,y)`：skimage `peak_local_max`，输入必须是**已 sigmoid 的概率图**（值域 [0,1]，传入原始 logits 直接报错，杜绝二次 sigmoid）；按（概率降序, 行, 列）排序，相邻峰/平局/边界行为完全确定；无检出返回 `(0,2)`。threshold/min_distance 只能在验证集上调参；
- `decode_semantic_mask(semantic_logits [3,H,W], uncertain_threshold=None, input_is_probability=False)`：默认对原始 logits 做数值稳定 softmax；若输入已经是概率，必须显式传 `input_is_probability=True`，此时校验值域和归一化并**直接使用，绝不二次 softmax**。返回 `{semantic_mask, confidence, uncertain}`；
- `separate_instances(semantic_mask, centers, max_assign_distance=None)`：每个中心一个实例 ID（1..N 与中心一一对应）；前景像素**仅在含中心的连通域内**按最近中心（精确欧氏 Voronoi）分配——**无中心的孤立前景碎片、超距像素保持 0（未分配）**，歧义不强行归属；两中心取整到同一像素时，后一个种子移到最近空闲像素，不覆盖 ID；返回 `{instance_mask [H,W] int64, cell_records(...), unassigned_foreground_pixels}`。交叉突起的归属仍需 Zhu 在显微图像上人工复核。

### 2.5 `src/predict.py` —— 推理全链路

#### 单图

```bash
python src/predict.py --image path/to/image.tif \
    --checkpoint checkpoints/best.pt --config configs/config.yaml \
    --output-dir results/predictions [--threshold 0.5 --min-distance 5 --channel 0 --device cpu]
```

`predict_image()` 返回：`semantic_mask [H,W]`、`instance_mask [H,W]`、`points [N,2] (x,y)`、`count`、`center_prob [H,W]`、`semantic_prob [3,H,W]`、`cell_records`、`provenance`。**count 是实例分离后的细胞数，不是裸峰数**。`data.channel` 只对实际多通道输入生效，单通道输入会显式警告并忽略，不再报错。

#### 全片 QPTIFF

```bash
python src/predict.py --slide path/to/slide.qptiff --checkpoint checkpoints/best.pt \
    --output-dir results/slide [--tile-size 512 --overlap 64]
```

**块内只做语义解码和中心提取，不做实例分配**；随后拼接语义掩膜、将中心换算为 level-0 坐标并全局去重，最后**用去重中心在整片上统一执行一次 `separate_instances`**。因此 `points/count`、`centers.csv`、`cell_records` 和实例掩膜 ID 始终一一对应。掩膜在所选 level 网格上拼接（重叠带保留先写入值）；超过 `inference.mask_pixel_budget` 时发出警告、不生成整片掩膜，CSV 仍按去重点导出且行数等于 count。`level>0` 时叠加图会先把点坐标换算回掩膜网格，不会跨 level 错位。

#### 输出文件（`save_predictions`）

| 文件 | 内容 |
| --- | --- |
| `centers.csv` | 每细胞一行：实例 ID、(x,y)、胞体/突起面积、uncertain 标志 |
| `semantic_mask.tif` | uint8 标签 0/1/2（无损） |
| `instance_mask.tif` | **uint32** 实例 ID（超过 255 个细胞也不丢 ID） |
| `overlay.png` | 检测点叠加图（大图自动缩为缩略图） |
| `summary.json` | count + provenance |

checkpoint 守卫：`load_trained_model` 校验 `output_heads == {semantic_classes: 3, center_channels: 1}`，**旧版单热图权重直接拒绝**，绝不静默加载。

## 3. 验证结果

| 验证项 | 结果 |
| --- | ---: |
| preprocessing 功能验证（含评审回归项） | 35/35 通过 |
| `train.py --self-test`（3 epoch，CPU） | 训练损失 2.17 → 1.85，checkpoint 保存/重载正常 |
| 仓库原有单元测试（tool/） | 25/25 通过 |
| 真实 Cellpose（cpsam，CPU）256×256 / 512×512 合成荧光图 | 检出 7 / 18 个实例 |
| 用 `evaluate.py` 对照合成真值（8 px 匹配半径） | P/R/F1 = 1.0，计数 MAE = 0 |
| **第二阶段：dataset/postprocessing/predict 端到端 + 评审回归检查** | **37/37 通过** |
| 真实 dataloader → `train.train()`（2 epoch，合成 BBBC042 式数据） | best/last.pt + history.json 正常产出 |
| `predict_image` / `predict_whole_slide` / CLI | 单图/全片契约完整；全片点数、CSV、记录、掩膜 ID 一致 |
| 第三轮评审专项回归 | 二次 softmax、同像素中心、全越界标注、单通道 channel、level>0、非物化路径均通过 |
| 旧版单热图 checkpoint | 按预期被拒绝 |

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
- **修改**：更新为 `output_channels: 4` 并注明含义；`train.build_model` 增加校验，配置与模型契约不一致时直接报错。第三阶段补齐 `data.channel / masks_dir / soma_radius_pixels / background_distance_pixels` 和 `inference.tile_size / overlap_pixels / level / mask_pixel_budget`，所有影响监督范围与全片推理的参数均显式可见。

### `docs/experiments.md`

- 追加五条实验记录，完整覆盖本分支工作：
  - **Experiment 003**：两个文件重建 + 端到端验证（27 项预处理检查、训练自测、25 个单测、真实 Cellpose、合成真值评估）；
  - **Experiment 003.v2**：第一轮评审修复（A1/A2/A4/A5/B1/B3/B5/B6/B7 + 2 项附加）与 35 项复验；
  - **Experiment 003.v3**：第二轮评审修复（check_dataset 扩展名、baseline 守卫、QYX 歧义文档化）；
  - **Experiment 004**：dataset / postprocessing / predict 全模块实现与 26 项端到端验证；
  - **Experiment 004.v2**：第三轮评审修复（全片统一实例分离、消除双重 softmax、channel 兼容、验证全覆盖等）与 37 项复验。

## 5. 遗留工作（不在本分支范围）

- `tool/check_dataset.py` 的 `validate_coordinates` 待确认 BBBC042 positions 格式后实现（`dataset.read_point_annotations` 已是容错解析器，可直接复用）；
- 评审中判定为"约定/范围差异"而未改的条目（A3 RGB 灰度化文档行为、B2 valid_mask 约定、B4 torch≥2.0、A6 tile_id 作用域），及两条低优先级遗留（A6 加固、B4 兜底），见 Experiment 003.v2 的处置表；
- 交叉突起的实例归属需 Zhu 在显微图像上人工复核（`separate_instances` 已保留未分配像素，不强行归属）；
- `inference.threshold` / `min_distance_pixels` 需在验证集上用 `tune_center_parameters()` 标定（当前沿用 config 默认值）。
