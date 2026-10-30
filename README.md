# 图表问答中的主动视觉观察与多轮强化学习

训练 Qwen2.5-VL 自主决定是否裁剪、观察哪个区域及何时回答，通过 B0/B1/B2 和观察方式对照检验主动选区的价值。

**代码与服务器运行流程已交付；真实 GPU 推理、训练及科学结果尚未验证。** 从 [服务器运行指南](docs/SERVER_RUN.md) 开始；验证范围见 [实施记录](docs/IMPLEMENTATION_STATUS.md) 和 [本地测试记录](reports/cpu-validation.json)。

目标环境：同一台 Linux 服务器上的 2 张 H200（用户报告每张约 143GB）、可新建 Conda 环境。模型选用 Qwen2.5-VL-7B-Instruct，使用 LoRA 更新语言部分；候选部署为 GPU 0 训练、GPU 1 生成。

## 本地检查

```bash
python -m pip install -e .
python scripts/run_cpu_tests.py
chart-rl --help
```

CPU 包仅依赖 Pillow。测试中的脚本化模型及框架替身只用于验证接口，不是模型实验结果。

## 服务器准备

```bash
conda env create -f environment.yml
conda activate chart-rl
bash scripts/install_server.sh
python scripts/download_assets.py --output-root data
python scripts/prepare_run.py --assets data/assets.json --output data_manifests/v1
python scripts/train.py --prepared data_manifests/v1 --setting B2 --output runs/B2-smoke --smoke
```

模型与数据下载使用 [项目配置](configs/project.json) 中固定的官方 revision。安装脚本保存实际 `pip freeze`；启动器先做环境检查，再运行真实多轮诊断。`runs/B2-smoke/g1-report.json` 通过后，按指南完成 pilot、B1/B2 训练与最终评估。模型、数据和运行产物保存在本机忽略目录中。

首个 smoke 会要求模型先裁剪一次，用于检查图片回传、损失 token、梯度、参数变化与更新后生成；它不作为自主选区效果实验。长训练要求相同初始化模型、相同代码内容的 G1 通过记录。

## 交付内容

- `src/chart_rl/`：数据与评分、真实裁剪、评估交互、训练插件接入及证据检查。
- `scripts/`：安装、下载、双卡训练、恢复、统一评估、结果汇总和离线案例查看。
- `plugins/chartqa_plugin.py`：ms-swift 裁剪调度器、答案奖励与训练审计注册。
- [实现说明](docs/DESIGN.md)：图像、采样 token、损失掩码和参数更新的对应关系。
- [第一天验收表](reports/day1-feasibility.md)、[实验报告模板](reports/experiment-report.md)：由服务器真实产物填写。

本项目提供实现与实验流程。训练产物、性能结论和简历中的量化结果，需要完成服务器实验后补充。

## 实验协议

| 编号 | 设置 |
| --- | --- |
| B0 | 共同初始化，未做 RL |
| B1 | 只看整图的 GRPO |
| B2 | 最多两次裁剪的多轮 GRPO |
| O0 / O1 | 普通整图 / 更高分辨率整图 |
| O2 / O3 | 整图加两个预先确定的固定 / 随机裁剪 |
| O4 | 模型自主决定是否裁剪及选区 |
| D0 | 人工有效区域，仅用于验证集诊断 |

O2/O3 是强制额外观察，O4 是自适应观察；共同上限不意味着实际成本相同。全部逐题结果均保留，成本缺失显示为 `null` 并报告覆盖率。

工具坐标指向首张实际显示图片，两次裁剪均映射到保留的原图。非法尝试消耗预算；达到两次后还有一次最终回答机会。工具响应返回真实图片。

## 答案评分

协议标识：`chartqa_pix2struct_trim_finite_v1`。参考 [Pix2Struct 的 ChartQA relaxed accuracy](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py)，使用 5% 相对数值容差及不区分大小写的文本匹配；额外去除首尾空白并拒绝非有限数值。

- 百分号转为小数：`50%` 可与 `0.5` 数值匹配，不能与 `50` 匹配。
- 零值沿用参考实现的文本回退：`0` 与 `0.0` 不匹配。
- 不去掉千分位、货币或其他单位，不做模糊文本匹配。
- 仅解析明确的 `<answer>...</answer>`，不抽取末尾数字；解析失败计错。
- 奖励仅为答案正确性 0/1，不奖励调用工具、格式或输出长度。

严格文本匹配同时报告。该协议不是 ChartQAPro 评分，也不应直接混同不同预处理口径的榜单结果。

## 数据与复现

遵循 [ChartQA 官方数据布局](https://github.com/vis-nlp/ChartQA)。跨划分图像重复按 test → val → train 优先保留；同图问题不会在自行划分中被拆开。重复问句但图像不同只记录。冲突标签会中止准备并要求检查。

数据、图片、模型、训练产物和本地路径记录放入忽略目录，不提交到代码仓库。服务器生成的 manifest、哈希、运行配置及逐题结果是实验追溯依据。去重不等于证明基座模型没有预训练污染。

## 开源依据

- [ms-swift v4.6.0](https://github.com/modelscope/ms-swift/tree/ad818a2cdfafaf4305ac40c4acd892a211978bc2)：复用 GRPO、生成 token/损失掩码管理及权重同步；本项目新增裁剪协议、规则奖励与数据/评估流程。
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)：基座模型和处理器。
- [ChartQA](https://github.com/vis-nlp/ChartQA)：图表问答数据。
- [Pix2Struct](https://github.com/google-research/pix2struct)：评分约定参考。

复用框架及模型的贡献与本项目实现分开说明。准确率、效率、训练成功及创新性结论均需以实际运行证据为准。
