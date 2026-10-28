# GFAP original ROI export validation

**技术导出核验：通过。标注人工复核：尚未完成。**

已核实20个ROI，每图1024×1024，来自GBM664_S5 / Scan1的QPTIFF level 0。
源文件：`D:\KP Lab\IHC slides\Competetion Slides\Real\GBM 好东西\GBM664_S5\Scan1\GBM664_S5_Scan1.qptiff`
SlideID：`GBM664_S5`；Identifier：`527e35fc-11f8-4f06-8dfc-390fcbbfb568`。
QPTIFF SHA-256：`f245419436556f741e3a22f247b7f7c6acf15e80d8f5462c1f095607f5d76fdc`。
TIFF像素尺度：X=0.249505974410452 µm/px，Y=0.249505974410452 µm/px。

## 核验结果

- PASS：20 ROI verified against annotation manifest and 20 CSV stems
- PASS：Crop metadata independently checked against export verification records and grid manifest
- PASS：Source QPTIFF SlideID, Identifier, channel metadata, shape and full-resolution level verified
- PASS：All 20 raw crops exactly equal existing quantitative source crops
- PASS：All 20 reconstructed RGB crops exactly equal original annotation images and unmarked reference views
- PASS：All PNG and raw-channel TIFF exports read back with exact pixel equality
- PASS：All 784 coordinates finite and in bounds; IDs contiguous; no identical duplicate points
- PASS：20 CSV copies byte-identical; statuses unchanged
- PASS：Existing annotation package, original RGB/raw tiles, overlays and uncertain panels unchanged
- PASS：Reference review material copied byte-identically; no image editing used to erase markers
- PASS：Image contact sheet uses unmarked RGB crops only; no text or markers overlaid

## 空间和显示变换

裁剪起点取自既有多通道TIFF的ImageDescription元数据，并与批量导出校验记录、tiles.csv及标注manifest交叉匹配。文件名中的坐标仅用于额外一致性检查。
原图无缩放、无旋转、无翻转；CSV为ROI内部0-based坐标，不加整片裁剪起点。所有输出图仍为1024×1024。
images/ 是无标记RGB PNG，与标注时看的原RGB TIFF逐像素一致。为了精确复现当时的显示，使用已有固定设置（不是本次新选的归一化）：DAPI显示范围0–255，Opal 620范围0–242，AF范围0–166；DAPI蓝色，Opal 620橙色(255,128,0)，AF不显示；gamma=1。浮点计算和舍入顺序记录于manifest及脚本。
raw_channels/ 是另附的3通道OME-TIFF，直接保存源QPTIFF level 0的uint8像素，未做显示归一化、通道删减、解混或背景扣除。通道依次为DAPI、Opal 620（用户说明为GFAP）、Sample AF。
源QPTIFF已记录OffsetAndShading预处理、尚未解混；此处原始数值指QPTIFF中现存数值，不是还原相机读数。
contact_sheet.png按manifest顺序由左到右、从上到下排20张，单图仅在总览中缩小到256×256；无文字或点叠加。总览不用于训练或坐标读取。

## 需要人工复核的内容

- 原有AI初标：confirmed=503，uncertain=281。两类均未获得第二位独立标注者复核。
- uncertain中包含数量未定的重叠区域，不能直接作为细胞数；应人工裁定细胞身份、中心位置、漏标、误标、重复及重叠区域。
- 第14张没有confirmed但有1个uncertain候选，不应当作已确认的零细胞样本。
- 原有逐图中文复核问题与uncertain坐标分别见provenance/original_annotation_summary.csv及original_annotation_uncertain_review.csv；对应叠加图和局部图原样复制到reference_review/。
- 当前CSV是点标注，可用于人工复核后的中心检测/计数评估；不包含轮廓或实例掩膜，因此不能直接完成U-Net/Cellpose的Dice、IoU或边界质量评估。若要评估分割，仍需独立补充像素/实例级掩膜。
- 没有生成新生物学标签，没有修改confirmed/uncertain，也没有虚构human_review_completed文件。

## 保留和交付范围

本次按当前明确要求保留原始文件名及4列CSV，未执行所附另一份重建草案中的重命名、CSV列扩展或覆盖图重绘。所有历史标注材料保持不变。
技术来源与像素对齐已核验；数据集仅在完成独立人工复核后才能作为固定的测试标签。
