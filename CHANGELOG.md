# Changelog

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
