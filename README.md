# A股量化分析与机器学习选股项目

本项目依据《东软金融量化分析期末实验项目考核要求》实现，包含：

1. 单只股票数据清洗、收益率与均线分析；
2. 基于几何布朗运动的蒙特卡洛模拟和99%下行风险分位；
3. 多股票特征工程、时间序列切分、机器学习二分类与候选股票筛选；
4. Streamlit Web 可视化平台；
5. 命令行批量生成图表、CSV和JSON分析结果。

## 目录结构

```text
股票分析/
├─ app.py
├─ data/
│  └─ A_share_quant_project_dataset.xlsx
├─ src/
│  ├─ data_loader.py
│  ├─ features.py
│  ├─ ml_selection.py
│  ├─ plots.py
│  └─ single_stock.py
├─ scripts/
│  └─ run_analysis.py
├─ outputs/
├─ requirements.txt
└─ README.md
```

## 安装与运行

建议使用 Python 3.10 或更高版本：

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 Streamlit 输出的本地地址即可进入可视化平台。

## 在 VS Code 中运行

项目已提供 `.vscode/settings.json`、`.vscode/launch.json` 和
`.vscode/tasks.json`，会自动使用已安装项目依赖的 `D:/ANACONDA/python.exe`。

操作方式：

1. 在 VS Code 中按 `Ctrl+Shift+P`，执行 `Python: Select Interpreter`；
2. 选择 `D:/ANACONDA/python.exe`；
3. 打开“运行和调试”面板，选择“启动 A 股量化分析平台”；
4. 点击绿色运行按钮，或按 `F5`；
5. 浏览器打开 `http://localhost:8501`。

不要直接运行 `python app.py`；Streamlit 应使用 `python -m streamlit run app.py` 启动。

第一次启动 Streamlit 可能会在终端显示 `Email:`，直接按回车留空即可。

## 批量生成实验素材

```powershell
python scripts/run_analysis.py --ticker 600519.SS --paths 5000 --threshold 0.55
```

脚本会在 `outputs/` 生成：

- 清洗后的行情数据；
- 重新计算的机器学习特征；
- 候选股票清单；
- 单股分析CSV；
- 单股、蒙特卡洛和多股票统计HTML交互图表；
- 可直接插入实验报告与答辩PPT的PNG图表；
- `summary_metrics.json` 汇总指标。

## 方法说明

- 移动平均线使用交易日窗口 MA20、MA52、MA252；
- 未来5日标签定义为 `Close[t+5] / Close[t] - 1 > 0`；
- 模型使用全局日期进行训练/验证/测试时间切分，避免未来数据泄漏；
- 99%风险分位使用模拟终值价格的下侧1%分位数，并计算相对当前价格的VaR；
- 涨停标记按主板约9.5%、创业板约19.5%、ST约4.5%的日收益阈值近似；
- 对一条OHLC逻辑异常记录进行清洗，并保留质量统计。
