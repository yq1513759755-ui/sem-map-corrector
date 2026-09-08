# SEM Map Corrector

实验室共享的 Zeiss SE2 SEM 十字标记定位与几何畸变校正工具。

本项目只维护旧版 **SE2 实心亮十字** 算法，不包含 InLens 浮雕风格的
自动识别。输入图像类型必须由操作者确认，不做自动风格切换。

## 一条命令下载

在 macOS 或 Linux 终端执行：

```bash
git clone --depth 1 https://github.com/yq1513759755-ui/Kitaev-Chain-Lab.git sem-map-corrector && cd sem-map-corrector && ./semcorr --help
```

在 Windows PowerShell 或 CMD 中执行：

```powershell
git clone --depth 1 https://github.com/yq1513759755-ui/Kitaev-Chain-Lab.git sem-map-corrector
cd sem-map-corrector
.\semcorr.cmd --help
```

下载后处理单张图像：

```bash
cd sem-map-corrector
./semcorr "/path/to/image.tif"
```

批量处理一个文件夹：

```bash
./semcorr --batch "/path/to/image_folder"
```

Windows 单张与批量处理：

```powershell
.\semcorr.cmd "C:\path\to\image.tif"
.\semcorr.cmd --batch "C:\path\to\image_folder"
```

## 最简单的调用方式

把整个 `sem-map-corrector` 文件夹复制到个人电脑后，在终端进入该目录：

```bash
./semcorr image.tif
./semcorr --batch image_folder
```

第一次运行会在项目内自动创建 `.venv`；只有缺少 NumPy、OpenCV 或
Matplotlib 时才会联网安装，之后仍使用相同命令，不会修改电脑的全局
Python 环境。如果实验室已有指定的 Python/Conda，首次运行建议指定它：

```bash
SEMCORR_PYTHON=/path/to/python ./semcorr image.tif
```

Windows 可用 `set SEMCORR_PYTHON=C:\path\to\python.exe`（CMD）或
`$env:SEMCORR_PYTHON='C:\path\to\python.exe'`（PowerShell）指定 Python。

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

输出包括校正 TIFF、中心坐标 CSV、中心标注图、检测图、残差图与 JSON
报告。程序不会修改输入图像。

## 结果解释

- JSON 中的四点单应残差为零只代表模型穿过四个输入点，不单独证明中心正确。
- 正式实验前应检查 `diagnostics/*_detection.png` 与中心标注图。
- 自动化测试使用合成图像，不随公开仓库分发实验图像、文件名、哈希或中心坐标。
- 四点单应残差不能代替人工查看检测图；用于正式实验前，请先确认绿色中心落在十字中心。

## 历史复现

原始单文件版本冻结在 `legacy/`。冻结文件不参与日常开发，不应编辑。
