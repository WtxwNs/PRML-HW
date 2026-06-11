# PRML 手写数字聚类课程设计

项目比较标准 K-means、改进 K-means++ 与 DBSCAN，并研究 Pixel、HOG、PCA
及自适应特征融合对手写数字聚类的影响。

## 环境

```powershell
uv sync
uv run pytest
uv run prml-benchmark --config configs/benchmark.yaml
```

快速验证：

```powershell
uv run prml-benchmark --config configs/smoke.yaml
```

运行最终 benchmark、消融、敏感性、稳定性实验并生成报告图表：

```powershell
uv run prml-experiments --config configs/final_experiments.yaml
```

当前最终方法为“倾斜校正 + 局部尺度谱图 + 模块度重启选择 + 邻域一致性细化”。
全量 Digits 数据集上 ACC 为 `0.9555`，10 个随机种子的平均 ACC 为
`0.9559 +/- 0.0002`。结果和图表位于 `outputs/final/`。

检查完整环境：

```powershell
.\scripts\check_env.ps1
```

Digits 数据集由 scikit-learn 内置提供，不需要单独下载，也不纳入最终提交文件。
实验结果默认写入 `outputs/`。

## 目录

- `src/prml_project/`：数据、特征、聚类器、指标和 benchmark
- `configs/`：正式实验及 smoke test 配置
- `tests/`：核心算法单元测试
- `outputs/`：实验表格与图片
- `SEU_PRML_Project_Template/`：XeLaTeX 报告

## TeX

推荐从项目根目录运行：

```powershell
.\scripts\build_report.ps1
```
