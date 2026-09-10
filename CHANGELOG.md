# Changelog

## 0.2.0

### 校正后中心标注（新增）

- 校正图 `<图像名>_corrected.tif` **就地**用红色小十字标出每个 mark 中心
  （只出一张图，打开即可见）。只做定位指示——不写坐标文字、不画圆圈，
  避免遮挡样品细节。校正图因此为 3 通道 RGB，标记为纯红 `#FF0000` 便于滤除。
- 标记尺寸：缺省臂长**固定 3 px、不随图像尺寸缩放**（常数
  `reporting.MARK_ARM_DEFAULT`），即总宽 7 px、由 **13 个红色像素**组成的
  正十字（7 横 + 7 竖 − 中心 1 个重叠）。臂长 N → 外接框 (2N+1)² px、
  红像素 4N+1 个。
- 新增 CLI 参数 `--mark-arm PX`：以**绝对像素**指定臂长，不再随图像缩放。
  `0` = 只画中心单像素（栅格图像的最小标记）、`1` = 3 px 十字、
  `N` = (2N+1) px 十字，单标记红像素 `4N+1` 个。
  另可通过 `correct_image(mark_arm=N)` 走库接口。
- 修复静默夹取：此前 `--mark-arm 0` 会被 `max(1, ...)` 悄悄变成 1，
  现在 0 表示单像素、负值显式报错。
- 线宽改为**恒为 1 px**（常数 `reporting.MARK_LINE_WIDTH`）。此前默认路径
  会随图像尺寸缩放到 2 px，于是同一张图"传了 `--mark-arm` 线宽 1、不传 2"，
  自相矛盾；现已去掉缩放，线宽与是否传 `arm_px` 无关。
- `draw_center_marks()` 增加越界裁剪，臂长超出画面时不再画出图像外。
- 标记画在 `self_check` **之后**——端到端自检必须在干净的校正图上重新检测。
- 曾短暂改为输出独立的 `<图像名>_corrected_marked.tif`（PNG 版曾放在
  `diagnostics/`），因用户打开 `corrected/` 找不到标记，改回单张内嵌方案。
- 新增 `diagnostics/<图像名>_centers_corrected.csv`（`id,x,y,source,unit`）
  与报告字段 `corrected_centers`，给出**校正图坐标系**下的 mark 中心。
  坐标优先取校正图上重新检测的结果（`source=self-check`），未复检到的
  格位回退到理想格位（`source=ideal-fallback`）。
- 报告 `schema_version` 由 1 升到 2（纯新增字段，旧字段未变）。

### 失败归因（改进）

- 标记数不足时不再建议改 `--grid`，改为**归因到 mark 十字是否残缺**：
  列出已接受标记的三项分数、指出缺失格位及其预测位置、逐项打印恢复门槛
  的通过情况（模板相关 / 臂角形状），并按失败的是哪一项给出成因提示
  （缺臂、被碎屑或套刻图形覆盖、刻蚀不完整、剂量不足、与亮图形粘连等）。
- 新增 `diagnostics/<图像名>_detection_failed.png`：绿圈 = 已接受、
  红叉 = 被图像门剔除、橙圈 = 缺失格位的几何预测位置及搜索范围。
- `recover_missing()` 增加可选 `diag` 出参，返回结构化的失败原因
  （`insufficient_points` / `row_split` / `no_hypothesis` / `gate_rejected`）；
  选优仍只看模板分，但三个分数在选优前一次算齐，失败时证据链完整。
  `diag` 为可选参数，旧调用方式不变。
- 新增 `estimate_spacing()`：用标记点两两最小距离估计网格间距，供诊断
  信息做尺度参照。

### 修复

- 中心坐标标签在四角布局下会互相遮挡（`_centers.png` 与新的
  `_corrected_marked.png` 同病）。新增标签避让布局：右→左→下→上依次试放，
  以压住标记/压住已放标签/越出画面的重叠面积为代价选最优。
- 新增输出文件加入 `OUTPUT_SUFFIXES`，批量处理时不会被当成输入图像重复读入。

### 检测器（修复）

- **面积门改为尺度自适应**，替换写死的 `MIN_AREA = 300`：
  `面积门 = max(MIN_AREA_FLOOR=30, 最大可信连通域面积 / MIN_AREA_RATIO=12)`。
  依据是本版图 mark 尺寸固定成族（大十字:小十字 面积比约 8–9），故"最大者 ÷
  常数"在任意倍率下都稳定落在小 mark 之下。旧写死阈值在降倍率时会先丢小十字、
  只留大十字，表现成"标记不齐全"，且极易被误判成"十字形状不完整"。
  实测 0115-01 缩到 80% 时旧规则只剩 1 个标记、自适应仍检全 4 个。
- `find_candidates()` / `detect_marks()` 增加可选 `diag` 出参；报告新增
  `detector` 字段（尺度基准 / 实际面积门 / 形态合格数 / 面积丢弃数），
  verbose 模式打印面积门。旧调用方式不变。
- 失败信息新增反向提示：若粗筛有结构因面积不足被丢弃，明确提示先怀疑
  **成像倍率**，而不是十字残缺。
- **零回归验证**：22 张真实图（20 成功 / 2 失败）逐位复核，标记位置偏移
  `0.00e+00`、自检 RMS 变化 `0.00e+00`、通过集完全一致；全批仅新增 1 个
  候选（0117-01），且被下游三道门拦截。

### 其他

- 移除 Windows 启动器 `semcorr.cmd` 与 CI 的 Windows 矩阵（实验室使用
  场景为 macOS + Jupyter；库代码保持跨平台，文件读写仍显式 UTF-8）。
- 回归测试 27 → 38 个：新增 mark 残缺归因、完整标记不误判、校正图内嵌红标与
  校正后 CSV、校正后中心构成正方形、红色小十字尺寸、`--mark-arm` 覆盖与负值
  拒绝、线宽恒为 1 px、面积门随尺度自适应、降倍率仍检全标记、固定阈值对照。

## 0.1.0

- 冻结原始 SE2 单文件程序。
- 建立 `src/` 布局的可安装项目与 `semcorr` 命令。
- 抽取检测、几何、质量检查、warp、I/O 与报告模块。
- 建立不依赖实验数据的合成回归测试。
- 增加 Windows PowerShell/CMD 原生启动器 `semcorr.cmd`。

## 0.1.1

- 项目从 Kitave-Chain 主仓库拆分为独立仓库（保留提交历史）。
- `correct_image()` / `process_single()` 增加 `verbose` 参数（默认 `True`
  保持 CLI 不变）；`verbose=False` 全程静默，供 notebook 与批量调用。
- 诊断图改用无副作用的 `Figure` API，库不再切换调用方的 matplotlib
  全局后端（notebook 内联绘图不再需要 `%matplotlib inline` 补救）。
- 演示图生成器移入 `semcorr.demo.make_demo_image()`，notebook 与测试
  共用同一实现。
- `semcorr_notebook.ipynb` 重写为精简版：环境自检、参数、静默运行、
  结果图、报告摘要、批量处理六个使用节。
- `correct_image()` 移除从未读取的 `diagnostics` 死参数。
- 移除 `legacy/` 冻结目录（旧版单文件程序仅保留在 git 历史中）。
- 增加 MIT LICENSE；CI 矩阵加入 Windows；README 补充 `--design`
  JSON 格式说明；包版本改为从 `semcorr.__version__` 动态读取。
