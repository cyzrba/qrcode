# QR Code Detection & Decoding — 项目文档

## 1. 项目概述

基于 **OpenCV + pyzbar** 的 QR 码检测与解码系统。针对 16 种不同拍摄条件下的 QR 码图像，实现了**分类别自适应预处理**管线，并支持**批量检测**与结果统计。

- **数据集**: `qrcodes/detection/` 下 16 个子目录，共 536 张图像
- **总体检测率**: 基线 51.9% → 自适应 **62.9%**（+11.0%）

---

## 2. 项目结构

```
qrcode/
├── main.py              # CLI 入口（单图 / 批量模式）
├── detector.py          # QR 检测与解码核心逻辑
├── preprocessor.py      # 16 类自适应图像预处理管线
├── pyproject.toml       # 依赖管理 (uv)
├── images/              # 测试图片
└── qrcodes/detection/   # 数据集（16 类子目录）
```

---

## 3. 依赖

| 包 | 用途 |
|---|---|
| `opencv-python` | 图像读写、灰度转换、预处理 |
| `pyzbar` | QR 码检测与解码 |
| `zbar` (系统库) | pyzbar 底层依赖，macOS 通过 `brew install zbar` 安装 |

> macOS 下 `detector.py` 会自动设置 `DYLD_LIBRARY_PATH` 指向 Homebrew 的 zbar 库路径。

---

## 4. 模块说明

### 4.1 `detector.py` — 检测核心

| 函数 | 说明 |
|---|---|
| `decode_qr(image_path)` | 基础检测：灰度转换 → pyzbar 解码 |
| `decode_qr_adaptive(image_path, category)` | 自适应检测：根据 category 调用预处理管线，依次尝试多个候选图像，返回首个成功结果；始终回退到原始灰度图 |
| `draw_results(image, results)` | 在图像上绘制绿色边界框和红色解码文本 |

返回结构：
```python
[{
    "data": "https://example.com",   # 解码内容
    "type": "QRCODE",                # 码类型
    "rect": {"x": ..., "y": ..., "w": ..., "h": ...},  # 矩形边界
    "polygon": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]    # 四边形顶点
}]
```

### 4.2 `preprocessor.py` — 预处理管线

核心思路：为每个类别生成**多个候选灰度图**，依次送入 pyzbar，直到有一个成功解码。

**共享工具函数：**

| 函数 | 作用 |
|---|---|
| `apply_clahe(gray, clip_limit)` | 自适应直方图均衡化，增强局部对比度 |
| `apply_unsharp_mask(gray, sigma, strength)` | 反锐化掩模，增强边缘/去模糊 |
| `apply_gamma(gray, gamma)` | Gamma 校正，调整亮度 |
| `apply_adaptive_threshold(gray, block_size, C)` | 自适应二值化 |
| `apply_morphological_close(gray, kernel_size)` | 形态学闭运算，填补断裂 |
| `_resize_gray(gray, max_dim)` | 按最大边缩放图像 |

### 4.3 `main.py` — CLI 入口

两种模式互斥：

```bash
# 单图模式
uv run main.py --image <path> [-c category] [-o output] [--no-display]

# 批量模式
uv run main.py --batch <dataset_dir> [-c category] [--output-dir dir]
```

---

## 5. 各类别预处理策略

| 类别 | 挑战 | 预处理方法 |
|---|---|---|
| **nominal** | 正常条件 | CLAHE |
| **blurred** | 运动/散焦模糊 | 反锐化掩模（多组参数） + CLAHE |
| **bright_spots** | 亮斑干扰 | 形态学 Blackhat 检测亮斑 → 减除 → CLAHE |
| **brightness** | 过曝/过暗 | Gamma 校正（0.5 暗图 + 2.0 亮图） + CLAHE |
| **close** | 特写/大尺寸 | 多尺度缩放（500/700/1000/1500px） + CLAHE |
| **curved** | 曲面变形 | CLAHE + 自适应二值化（多组 block_size） |
| **damaged** | 物理损伤 | 多尺度 CLAHE + 自适应二值化 + 形态学闭运算 |
| **glare** | 反光/眩光 | Inpainting 去高光 → 多尺度 + CLAHE |
| **high_version** | 高版本密集码 | 多尺度缩放（400~1500px） + CLAHE |
| **lots** | 多码共存 | CLAHE（pyzbar 原生支持多码检测） |
| **monitor** | 屏幕截图 | 多尺度缩放（500/800/1000/1500px） + CLAHE |
| **noncompliant** | 非标准 QR | CLAHE + 自适应二值化（多组参数） |
| **pathological** | 极端情况 | 原始灰度 + CLAHE + 反锐化 + 自适应二值化 |
| **perspective** | 透视变形 | CLAHE + 自适应二值化 |
| **rotations** | 旋转 | CLAHE + 原始灰度 |
| **shadows** | 阴影遮挡 | CLAHE（clip_limit=3.0） |

**关键发现**: pyzbar 在图像最大边 **500~1500px** 时检测效果最佳。大尺寸图像（如 3024×4032）必须缩放后才能识别。

---

## 6. 使用示例

```bash
# 单张图片检测
uv run main.py --image photo.jpg --no-display

# 指定类别（启用自适应预处理）
uv run main.py --image photo.jpg -c glare --no-display

# 批量检测整个数据集
uv run main.py --batch qrcodes/detection/

# 批量检测单个类别
uv run main.py --batch qrcodes/detection/ -c monitor

# 批量检测并保存标注结果
uv run main.py --batch qrcodes/detection/ --output-dir results/
```

---

## 7. 测试结果

### 7.1 基线 vs 自适应 对比

| 类别 | 基线 | 自适应 | 提升 |
|---|---|---|---|
| monitor | 0.0% | **82.4%** | **+82.4** |
| close | 12.5% | **67.5%** | **+55.0** |
| glare | 40.0% | **72.0%** | **+32.0** |
| brightness | 64.3% | **89.3%** | **+25.0** |
| bright_spots | 40.6% | **62.5%** | **+21.9** |
| pathological | 65.2% | **87.0%** | **+21.8** |
| high_version | 21.2% | **36.4%** | **+15.2** |
| damaged | 21.6% | **35.1%** | **+13.5** |
| perspective | 42.9% | **51.4%** | **+8.5** |
| noncompliant | 62.5% | **68.8%** | **+6.3** |
| blurred | 46.7% | **53.3%** | **+6.6** |
| curved | 42.0% | **46.0%** | **+4.0** |
| nominal | 73.8% | 73.8% | 0 |
| rotations | 61.4% | 61.4% | 0 |
| shadows | 85.7% | 85.7% | 0 |
| lots | 100.0% | 100.0% | 0 |
| **总计** | **51.9%** | **62.9%** | **+11.0** |

### 7.2 各类别详细数据

| 类别 | 检测数 | 总数 | 检测率 |
|---|---|---|---|
| blurred | 24 | 45 | 53.3% |
| bright_spots | 20 | 32 | 62.5% |
| brightness | 25 | 28 | 89.3% |
| close | 27 | 40 | 67.5% |
| curved | 23 | 50 | 46.0% |
| damaged | 13 | 37 | 35.1% |
| glare | 36 | 50 | 72.0% |
| high_version | 12 | 33 | 36.4% |
| lots | 7 | 7 | 100.0% |
| monitor | 14 | 17 | 82.4% |
| nominal | 48 | 65 | 73.8% |
| noncompliant | 11 | 16 | 68.8% |
| pathological | 20 | 23 | 87.0% |
| perspective | 18 | 35 | 51.4% |
| rotations | 27 | 44 | 61.4% |
| shadows | 12 | 14 | 85.7% |
| **TOTAL** | **337** | **536** | **62.9%** |

---

## 8. 仍存在的难点

| 类别 | 检测率 | 原因分析 |
|---|---|---|
| high_version | 36.4% | 高版本 QR 码模块密集，pyzbar 解码能力有限 |
| damaged | 35.1% | 物理损伤导致定位图案/数据区丢失，超出纠错能力 |
| curved | 46.0% | 曲面导致严重几何畸变，需更复杂的展开算法 |
| perspective | 51.4% | 大角度透视需单应性变换校正 |

这些类别需要引入更强的手段（如深度学习检测器、透视变换估计、QR 码几何展开等）才能进一步提升。
