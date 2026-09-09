# Changelog

## Unreleased

- 移除 Windows 启动器 `semcorr.cmd` 与 CI 的 Windows 矩阵（实验室使用
  场景为 macOS + Jupyter；库代码保持跨平台，文件读写仍显式 UTF-8）。

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
