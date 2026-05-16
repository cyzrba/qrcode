# QR码检测 - 各类别预处理方式总结

## 基础工具函数

| 函数 | 作用 | 关键参数 |
|------|------|----------|
| `apply_clahe` | 对比度受限自适应直方图均衡化，增强局部对比度 | `clip_limit=2.0`, `grid_size=(8,8)` |
| `apply_unsharp_mask` | 反锐化掩模，增强边缘细节（去模糊） | `sigma`, `strength` |
| `apply_gamma` | Gamma 校正，调整亮度 | `gamma<1` 提亮，`gamma>1` 压暗 |
| `apply_adaptive_threshold` | 自适应阈值二值化 | `block_size`, `C` |
| `apply_morphological_close` | 形态学闭运算，填补小孔/连接断裂 | `kernel_size` |
| `_resize_gray` | 按最大边长等比缩放 | `max_dim` |

## 检测流程

1. 读取图片 → 转灰度
2. 根据 `category` 选择对应的预处理函数，生成多个候选图（candidates）
3. 依次对每个候选图调用 `pyzbar.decode()` 尝试解码
4. 任一候选图解码成功即返回结果

## 各类别处理策略

### 1. nominal（正常场景）
**策略：** CLAHE 增强对比度
- 候选图：`CLAHE(gray)`

### 2. blurred（模糊图像）
**策略：** 反锐化掩模锐化 + CLAHE 增强
- 候选图：
  - `CLAHE(unsharp_mask(gray, σ=2.0, strength=2.0))` — 强锐化
  - `CLAHE(unsharp_mask(gray, σ=1.0, strength=1.5))` — 中等锐化

### 3. bright_spots（亮斑/高光斑点）
**策略：** Blackhat 形态学运算消除亮斑 + CLAHE
- 候选图：
  - `CLAHE(gray - blackhat(gray))` — 亮斑校正后增强
  - `CLAHE(gray)` — 原始灰度增强（兜底）

### 4. brightness（过亮/过暗）
**策略：** Gamma 校正生成不同亮度版本
- 候选图：
  - `CLAHE(gamma(0.5))` — 暗图提亮（γ=0.5）
  - `CLAHE(gamma(2.0))` — 亮图压暗（γ=2.0）
  - `CLAHE(gray)` — 原始灰度增强

### 5. close（近距离拍摄）
**策略：** 多尺度缩放 + CLAHE
- 候选图：在 `[500, 700, 1000, 1500]` 四个尺度下，分别生成原始缩放和 CLAHE 增强版本（共 8 张）

### 6. curved（弯曲/曲面）
**策略：** 强 CLAHE + 自适应阈值二值化
- 候选图：
  - `CLAHE(gray, clip_limit=3.0)` — 强对比度增强
  - `adaptive_threshold(clahe, 21, 3)` — 中等块大小二值化
  - `adaptive_threshold(clahe, 31, 5)` — 大块二值化

### 7. damaged（损坏/残缺）
**策略：** 多尺度 + 自适应阈值 + 形态学闭运算（最复杂）
- 候选图：
  - CLAHE 后在 `[800, 1000, 500, 1500]` 尺度缩放（4 张）
  - CLAHE + 缩放后自适应阈值 `[800, 1000, 500]`（3 张）
  - 形态学闭运算（填补损坏区域）
  - 闭运算 + 自适应阈值

### 8. glare（反光/眩光）
**策略：** Inpaint 修复高光区域 + 多尺度
- 候选图：
  - 原始灰度多尺度 `[800, 1000, 500, 1500]`（4 张）
  - CLAHE 灰度缩放 `[800, 1000, 500]`（3 张）
  - Inpaint 修复 + CLAHE + 缩放 `[800, 1000, 500]`（3 张）
- **核心思路：** 用 `cv2.inpaint` 填充高光区域（>240 的像素），再用 CLAHE 增强

### 9. high_version（高版本QR码，信息密度高）
**策略：** 多尺度缩放（优先大尺寸）+ CLAHE
- 候选图：`[1500, 1200, 800, 600, 400]` 尺度缩放 + `CLAHE(gray)`
- **特点：** 优先使用大尺寸保留细节

### 10. lots（大量QR码）
**策略：** 简单 CLAHE 增强
- 候选图：`CLAHE(gray)`

### 11. monitor（屏幕拍摄）
**策略：** 多尺度 + 强 CLAHE
- 候选图：在 `[500, 800, 1000, 1500]` 尺度下，分别生成原始缩放和 `CLAHE(clip_limit=3.0)` 版本（共 8 张）

### 12. noncompliant（不合规/非标准QR码）
**策略：** CLAHE + 多参数自适应阈值
- 候选图：
  - `CLAHE(gray)`
  - `adaptive_threshold(clahe, 11, 2)` — 小块精细二值化
  - `adaptive_threshold(clahe, 21, 3)` — 中等二值化
  - `adaptive_threshold(clahe, 31, 5)` — 大块粗略二值化

### 13. pathological（极端病态场景）
**策略：** 多种增强手段组合（最全面）
- 候选图：
  - 原始灰度 `gray`
  - `CLAHE(gray, clip_limit=3.0)` — 强对比度
  - `CLAHE(unsharp_mask(gray, σ=1.5, strength=2.0))` — 锐化 + 增强
  - `adaptive_threshold(clahe, 11, 2)` — 小块二值化
  - `adaptive_threshold(clahe, 21, 3)` — 中等二值化

### 14. perspective（透视变换/倾斜拍摄）
**策略：** CLAHE + 自适应阈值
- 候选图：
  - `CLAHE(gray)`
  - `adaptive_threshold(clahe, 21, 3)`
  - `adaptive_threshold(clahe, 31, 5)`

### 15. rotations（旋转）
**策略：** CLAHE + 原始灰度
- 候选图：`CLAHE(gray)`, `gray`

### 16. shadows（阴影）
**策略：** 强 CLAHE 补偿阴影
- 候选图：`CLAHE(gray, clip_limit=3.0)`

## 技术手段使用频次统计

| 技术手段 | 使用类别数 | 适用场景 |
|----------|-----------|----------|
| CLAHE | **全部 16 类** | 通用对比度增强，几乎每类都用 |
| 多尺度缩放 | 6 类 | close, damaged, glare, high_version, monitor, |
| 自适应阈值 | 6 类 | curved, damaged, noncompliant, pathological, perspective |
| 反锐化掩模 | 2 类 | blurred, pathological |
| Gamma 校正 | 1 类 | brightness |
| 形态学 Blackhat | 1 类 | bright_spots |
| 形态学闭运算 | 1 类 | damaged |
| Inpaint 修复 | 1 类 | glare |

## 设计思路总结

1. **分类驱动**：不同图像退化类型需要不同的预处理策略，避免一刀切
2. **多候选竞争**：每个类别生成多个候选图，按序尝试解码，命中即返回
3. **由轻到重**：候选图按处理强度递增排列，简单场景快速命中，复杂场景逐步尝试
4. **灰度兜底**：`preprocess()` 函数确保原始灰度图始终作为最后的候选
