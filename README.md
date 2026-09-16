# SEM Map Corrector

[![CI](https://github.com/yq1513759755-ui/sem-map-corrector/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yq1513759755-ui/sem-map-corrector/actions/workflows/ci.yml)

面向实验室的 **SE2 SEM 十字标记定位、图像几何校正与 AutoCAD 数值贴图工具**。
提供原图批量处理到 CAD 贴图包的一键工作流，减少人工点击十字中心和逐张调整比例的操作。

**当前版本：0.3.1。** 支持 SE2 实心亮十字；不自动识别 InLens 浮雕风格。
CAD 贴图使用人工提供的位置标注，不从重复标记阵列猜测绝对坐标。

## 主要功能

- 自动裁去底部 SEM 参数栏及白色边框，裁下的参数栏另存备查。
- 模板检测结合四臂边缘中轴交点定位，提供亚像素中心、几何校正和校正后复检。
- 从文件名读取左下十字的版图坐标，生成 AutoCAD 贴图脚本、图像副本及数值参数。
- 统一记录通过、需复核和失败结果；本次失败图片不会借用历史成功报告进入贴图包。
- 支持命令行、Python API 和 Jupyter 校正；输入原图不被修改。

## 快速开始

### 1. 下载

```bash
git clone --depth 1 https://github.com/yq1513759755-ui/sem-map-corrector.git
cd sem-map-corrector
```

需要 Python 3.11 或以上版本。macOS / Linux 下的 `./semcorr` 首次运行会在项目内
创建 `.venv`；缺少 NumPy、OpenCV 或 Matplotlib 时安装依赖，后续复用该环境。
如需指定 Python：

```bash
SEMCORR_PYTHON=/path/to/python ./semcorr --help
```

### 2. 给原图标注左下坐标

例如 `sample(3.5,2).tif`：括号内表示**图中左下十字中心**应处于
版图 `(350,200) µm`，即括号数值乘以 100。

CAD 流程约定：四标记 2×2 网格，默认相邻间距 50 µm，图像向右为版图 +X、
向上为 +Y，版图 1 个绘图单位代表 1 µm。相同外观的重复标记不能提供绝对位置，
因此文件名标注需由操作者确认。

### 3. 一条命令完成校正与贴图包生成

```bash
./semcorr --batch "/path/to/annotated_images" --cad
```

结果写入该图片目录的 `corrected/`：

```text
corrected/
├── *_corrected.tif       校正图，包含红色中心标记
├── diagnostics/          定位诊断、坐标、参数栏与逐图报告
├── workflow_summary.json 批次状态及未导出原因
└── cad/
    ├── sem_map.lsp       AutoCAD 贴图程序
    ├── attach_all.scr    批量运行入口
    ├── images/           贴图使用的图像副本
    ├── cad_params.csv    插入点、比例、角度及残差
    └── cad_manifest.json 坐标约定、输入哈希及导出记录
```

### 4. 在 AutoCAD 中贴图

在版图副本的模型空间、命令空闲状态操作：

1. `APPLOAD` 加载本批次 `corrected/cad/sem_map.lsp`。
2. `SEMMAPONE` 试贴第一张，检查位置、方向和尺寸。
3. `SEMMAP` 批量贴图，`SEMMAPCHECK` 检查实际图像变换。
4. 核对后另存 DWG，并保留 `cad/images/`；图像是外部参照。

每张图片使用独立 `SEM_` 图层；同一贴图包重复运行时检查已有图像，避免重复插入。
程序不自动打开、控制或保存 AutoCAD 图形。配准残差不是实际曝光套刻精度。

## 其他常用入口

只校正单张或整批图像（无需坐标文件名）：

```bash
./semcorr image.tif
./semcorr --batch image_folder
```

已有校正结果时，仅生成贴图包：

```bash
.venv/bin/python scripts/make_cad_align.py "/path/to/annotated_images"
```

改变输出目录、标记物理间距或 CAD 单点误差门槛：

```bash
./semcorr --batch image_folder --cad --outdir output_folder \
  --pitch-um 50 --max-residual-um 0.05
```

详细参数见 `./semcorr --help`。一键流程只接受通过质量检查的结果；有未导出图片时
返回码为 1，但其他合格图片仍生成贴图包。原因见 `workflow_summary.json`。

## 标准安装

建议使用 Python 3.11–3.13 的独立虚拟环境：

```bash
python -m pip install .
```

开发安装：

```bash
python -m pip install -e '.[test]'
```

安装完成后，可以从任意目录直接调用：

单张图像：

```bash
semcorr image.tif
```

批量处理：

```bash
semcorr --batch image_folder
```

六标记版图：

```bash
semcorr image.tif --grid 2x3
```

### 高级校正坐标（可选，`--design`）

默认从检测点推断正方形理想网格。`--design` 可指定 M1..Mn 的目标坐标，
但当前实现会直接将其作为输出栅格坐标，输出尺寸仍沿用原图，尚不自动换算
物理单位、原点或画布范围。坐标应位于输出画面内。

**CAD 贴图请使用文件名括号坐标和 `--cad`，不要把芯片绝对微米坐标直接传给
`--design`。** 两个参数不能在一键流程中同时使用。

```bash
semcorr image.tif --design design.json
```

`design.json` 格式——键为 mark 编号 `M1..Mn`（行优先，与输出报告一致），
值为该 mark 的设计坐标 `[x, y]`，缺少任何键会报错：

```json
{
  "M1": [100.0, 100.0],
  "M2": [300.0, 100.0],
  "M3": [100.0, 300.0],
  "M4": [300.0, 300.0]
}
```

报告中的 `ideal` 为提供的目标坐标；拟合残差和 `*_centers.csv` 仍在原图像素
坐标系中。`--design` 路径尚存在单位命名限制，不作为物理坐标标定接口。

## 自动裁掉底部参数栏

Zeiss（GeminiSEM）导出的原图底部带一条深色参数栏（Mag / WD / EHT /
Signal A / Date / Time …）。**这类原图可以直接丢进来，程序会自动检测并裁掉**，
不需要事先手工裁切：

```bash
./semcorr "/path/to/raw_from_sem.tif"      # 自动裁，无需额外参数
./semcorr --batch image_folder             # 批量同样自动裁
./semcorr raw.tif --keep-info-bar          # 需要保留参数栏时（几乎不用）
```

裁下的条带会另存为 `diagnostics/<图像名>_infobar.png` —— **实验参数不丢**，
随时可以回看或做 OCR。报告里的 `info_bar` 字段记录裁切位置：

```json
"info_bar": {"top": 707, "bottom": 762, "height": 56, "frac": 0.073,
             "step": 255.0, "bg_pct": 0.0}
```

### 边框与固定高度

同一种导出格式的参数栏高度可以一致，但已裁剪图、不同分辨率和导出格式
不能共用一个无条件固定裁切值。程序始终在标记检测前裁切信息栏。
0.2.1 会检查底部各个暗行连续段，跳过白色下边框外的一行黑像素；
裁切同时包含上方最多 4 行近白通栏分隔线，不会留下两行白边。
`top` 为实际裁切位置，`dark_top` 为黑底起始行，`removed_rows` 为从图中
实际去掉的总行数（包含上下边框及尾行）。重复处理已裁图不会再固定削去一截。

### 判据

四条**同时**成立才认定（保守优先：误裁比不裁更糟）：

| # | 判据 | 说明 |
|---|---|---|
| 1 | 底部存在连续深色行 | 行内**近黑像素占比 > 25%**。用占比而非中位数/均值 —— 图越窄白字占行宽比例越大，中位数会被抬到 43+ 而打断连续段 |
| 2 | 条带贴底 | 下沿距画面底部 ≤ 3% 图高（容忍带窗口边缘的截图） |
| 3 | 高度占 1.5%–20% | 上限同时挡掉"整幅图都暗"这种把全图判成条带的退化情形 |
| 4 | 底色近乎纯黑 + 上沿陡降 + 栏内有近白文字 | 三条一起把"样品本身的暗区"排除掉 |

阈值集中在 `src/semcorr/infobar.py` 顶部，可单独调整。判据只针对**底部**
信息栏；若你的 SEM 把参数栏放在别的位置，需要改 `detect_info_bar()`。

输出包括校正 TIFF（自带红色中心标记）、校正后中心坐标 CSV、中心标注图、
检测图、残差图与 JSON 报告。程序不会修改输入图像。

### 中心定位与精度（0.2.1）

模板匹配先确定标记邻域，随后在十字四个臂上逐条测量两侧的半对比度边缘。
用边缘中点拟合横臂、竖臂中轴，两条中轴的交点作为最终中心。这样不把
十字臂长不一致、模板宽度不匹配或局部亮度差导致的相关峰偏移当作中心。
原图检测、缺失恢复和校正图复检都使用同一精化过程。

两侧臂都必须提供足够的有效剖面；剖面过宽、污染严重或拟合不稳定时不强行
精化。报告 `marks[*].center_refinement` 和
`self_check.marks[*].center_refinement` 记录定位方法、剖面数量及边缘拟合误差。
复检中心只有模板定位、没有四臂边缘证据时，质量状态为 `WARN_REVIEW`，
批量汇总会单独列出“需复核”。`PASS` 仍不是绝对物理定位精度的证明。

红标绘制在最接近中心的整数像素上；CSV/JSON 保留亚像素坐标。

### 校正后中心标记

`<图像名>_corrected.tif` **本身就带红色小十字**，标出每个 mark 的中心——
打开校正图就能看到，不需要另找文件。标记只做定位指示：不写坐标文字、
不画圆圈，尺寸很小，避免遮挡样品细节。

尺寸：缺省臂长固定 **3 px**（**不随图像尺寸缩放**），即总宽 7 px、由
**13 个红色像素**组成的正十字（7 横 + 7 竖 − 中心 1 个重叠）：

```
      █
      █
      █
  █ █ █ █ █ █ █
      █
      █
      █
```

需要别的尺寸用 `--mark-arm` 指定**绝对像素**的臂长：

| `--mark-arm` | 标记外接框 | 红色像素 | 形状 |
|---|---|---|---|
| `0` | 1 × 1 px | 1 | 单点（最小） |
| `1` | 3 × 3 px | 5 | 小十字 |
| `2` | 5 × 5 px | 9 | 小十字 |
| `3`（缺省） | 7 × 7 px | 13 | 十字 |
| `N` | (2N+1) × (2N+1) px | 4N+1 | 十字 |

```bash
./semcorr image.tif                       # 缺省：13 像素十字
./semcorr image.tif --mark-arm 1          # 5 像素十字
./semcorr --batch image_folder --mark-arm 0   # 单像素（最小）
```

`--mark-arm` **只影响臂长**；线宽由常数 `reporting.MARK_LINE_WIDTH` 控制，
恒为 1 px —— 这已是栅格图像的**线宽下限**，不可能更细。想让标记"更淡"只能
改颜色或加抗锯齿（那是变淡而不是变细）。负值会显式报错，不会被静默夹到
最小值。

> 校正图因此是 3 通道 RGB（标记为纯红 `#FF0000`，便于程序化滤除）。若后续
> 分析需要不含标记的版本，把 `pipeline.py` 里的
> `write_image(out_img, draw_center_marks(corrected, corr_centers))`
> 改回 `write_image(out_img, corrected)` 即可。
>
> 标记画在 `self_check` **之后**——端到端自检必须在干净的校正图上重新
> 检测，红线不能干扰它。

坐标不进图，以机器可读形式写在 `diagnostics/<图像名>_centers_corrected.csv`
与报告 `corrected_centers` 字段中：

```csv
id,x,y,source,unit
M1,100.045,78.432,self-check,px
```

- 坐标取校正图上**重新检测**的结果（`source=self-check`），即校正后的真实
  测量值；个别未复检到的格位回退到理想格位（`source=ideal-fallback`）。
- 单位与报告一致：默认 px，传了 `--design` 则为设计单位。
- 这是后续定位实验应当引用的坐标系 —— 原图坐标仍在
  `diagnostics/<图像名>_centers.csv` 中。

### 标记不齐全时的失败诊断

标记数不足且网格预测恢复失败时，程序不会建议你改 `--grid`，而是**直接归因**：
列出已接受的标记及其模板/对称/形状分，指出缺失的是哪个格位、它应该在哪，
以及该位置逐项的门槛判定。例如：

```
按其余标记的网格几何，缺的是 M4，它应当位于 (617.8, 606.9) px。
该位置的局部模板搜索与几何验证结果：
   模板相关  0.674   （门槛 0.55）   通过
   臂角形状  0.278   （门槛 0.35）   未通过 ←
```

同时落一张 `diagnostics/<图像名>_detection_failed.png`：绿圈 = 已接受、
红叉 = 被图像门剔除、橙圈 = 缺失格位的几何预测位置（含搜索范围），
放大到橙色圈即可判断该处十字的四个臂是否完整。

### 检测器的自适应面积门

粗筛的面积下限**不写死绝对像素**，而是按图内 mark 的尺度自适应：

```
面积门 = max(MIN_AREA_FLOOR, 最大可信连通域面积 / MIN_AREA_RATIO)
       = max(30 px²,              尺度基准 / 12)
```

依据是本版图的 mark 尺寸是**固定的一族**（大十字 : 小十字 面积比约 8–9 倍），
所以"最大者 ÷ 常数"在任意倍率下都稳定落在小 mark 之下。写死绝对像素
（旧版为 `MIN_AREA = 300`）会在降倍率时先丢掉小十字、只留下大十字，表现就是
"标记不齐全"，且极易被误判成"十字形状不完整"。

实测（0115-01 逐级缩小）：

| 缩放 | 小十字面积 | 旧固定 300 | 自适应 |
|---|---|---|---|
| 100% | 393 px² | 4 个 | 4 个 |
| 80% | 241 px² | **1 个** | 4 个 |
| 60% | 140 px² | **1 个** | 4 个 |
| 40% | 58 px² | **1 个** | 4 个 |

报告中的 `detector` 字段会记录每次运行的尺度基准、实际面积门与丢弃数：

```json
"detector": {"anchor_area_px2": 2699.0, "min_area_px2": 224.9,
             "n_plausible": 42, "n_candidates": 7, "n_dropped_by_area": 35}
```

失败信息里若出现"⚠ 有 N 个亮结构因面积不足被丢弃"，请先确认**成像倍率**
是否偏低，而不是直接怀疑十字残缺。

## 一条命令：校正并生成 AutoCAD 贴图包

带左下坐标标注的原图可以直接运行：

```bash
./semcorr --batch /path/to/annotated_images --cad
```

程序依次裁信息栏、定位中心、校正、复检，再生成 `corrected/cad/sem_map.lsp`。
`corrected/workflow_summary.json` 统一记录可贴图、校正失败、需复核及 CAD 检查
未通过的图片和原因。本次失败或需复核的图片不会复用以前的成功报告。
有图片未导出时返回码为 1，但其余合格图片仍正常生成贴图包。

在 AutoCAD 中 APPLOAD 加载本批次 `sem_map.lsp`，先 `SEMMAPONE`，核对后再
`SEMMAP`。本命令只生成文件，不会打开或控制 AutoCAD。

- 原有不带 `--cad` 的单图、批量校正用法保持不变。
- 一键流程限 `--grid 2x2`，不能同时使用 `--design`；绝对坐标来自文件名括号。
- `--outdir DIR` 指定整个输出目录，贴图包在 `DIR/cad/`。
- 可选 `--anchor-overrides FILE`、`--pitch-um 50`、`--max-residual-um 0.05`。
- 已有校正结果仍可用下文的独立 CAD 导出脚本，避免重新校正。

## AutoCAD 数值贴图（0.3.0）

文件名末尾的 `(x,y)` 表示**图中左下十字 M3** 的设计坐标，数值乘以
100 转为 µm。例如 `0303-02(3.5,2).tif` 的左下锚点为 `(350,200) µm`。
此模式不识别数字、不依赖大十字，也不从原有四位编号猜坐标。
图像向右对应 CAD +X，图像向上对应 CAD +Y；默认相邻间距为 50 µm。

先按原命令生成校正图，再运行：

```bash
.venv/bin/python scripts/make_cad_align.py /path/to/annotated_images
```

结果在该批次的 `corrected/cad/`：

- `sem_map.lsp`：在 AutoCAD 用 APPLOAD 加载，先输入 `SEMMAPONE` 试贴第一张，确认后 `SEMMAP` 批量贴图。
- `attach_all.scr`：也可用 SCRIPT 加载这一文件。
- `cad_params.csv`、`cad_manifest.json`：插入点、每像素比例、角度、逐点残差、跳过原因和哈希。
- `images/`：贴图用的校正图副本。请保留整个文件夹，DWG 中图像是外部参照。

在版图**副本的模型空间**运行，1 个绘图单位代表 1 µm。先取消尚未完成的
文件选择或其他命令，再运行脚本。每张图使用独立的 `SEM_` 图层；同一批结果
重复运行会跳过已经放置且变换一致的图像。`SEMMAPCHECK` 可核查 CAD 中实际
IMAGE 的尺寸与变换。脚本不会自动保存或覆盖当前图形。

左下锚点严格固定；其余三个点联合拟合旋转和等比缩放。导出要求质量 PASS、
四个中心均为边缘复检结果、原图哈希一致、单位为 px，且最大单点配准残差不超过
0.05 µm。失败图像只记录原因，不导出虚构坐标。配准残差不是曝光套刻精度。

为了避免 DPI/INSUNITS 引入比例错误，先附着图像，再显式设置 IMAGE 的
DXF 10/11/12（插入点和两个每像素向量）；不把 µm/px 直接当作附着缩放倍数。
像素中心转换采用 `(x+0.5, H-y-0.5)`，包含相对外边界的半像素偏移。

原图重命名后，可用输入 SHA256 重新关联唯一的已有校正报告，无需仅因改名重算。
错写的坐标不会自动猜测；可修正文件名，或在批次目录写 `cad_anchor_overrides.json`：

```json
{"image_name_with_annotation": [3.5, 2]}
```

覆盖值同样以 100 µm 为单位。可选参数：`--anchor-overrides FILE`、
`--pitch-um 50`、`--max-residual-um 0.05`、`--outdir DIR`。

## Jupyter 交互版

不想用命令行的用户可以打开 `semcorr_notebook.ipynb`：与 CLI 完全等价
（同一个 `semcorr.correct_image()` API），四张诊断图内联显示、报告摘要
直接成表、支持批量处理。改第 1 节参数后 Run All 即可；内核需要
`numpy / opencv-python-headless / matplotlib`。

## 结果解释

- JSON 中的四点单应残差为零只代表模型穿过四个输入点，不单独证明中心正确。
- 正式实验前应检查 `diagnostics/*_detection.png` 与中心标注图。
- **默认 2x2（4 个 mark）时，`homography.rms_px` 与 `used_model_rms_px` 恒为
  ~1e-5，是四点单应的平凡零残差，没有质量含义**；有诊断价值的是
  `affine.rms_px`（体现非仿射/透视成分）与 `self_check.rms`（校正后端到端
  复检）。留一验证在 4 点下会返回"不适用"（无冗余可校验）。
- **理想网格是从检测点反推的**（除非传 `--design`），所以 `self_check` 只证明
  模型自洽，不证明那 4 个点就是设计意图的 mark —— CAD 定位请用
  `--cad`，结合人工左下坐标和已知物理间距标定。校正图本身仍以像素表示，
  跨图比较物理长度时须使用各图的 µm/px 比例。
- 自动化测试使用合成图像，不随公开仓库分发实验图像、文件名、哈希或中心坐标。
- 四点单应残差不能代替人工查看检测图；用于正式实验前，请先确认红色中心落在十字中心。
