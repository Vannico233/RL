# 服务器运行指南

本指南面向同机 2 张 H200、可创建 Conda 环境的 Linux 服务器。显存按用户报告的每张约 143GB 规划；实际卡型、可用显存、驱动及 CUDA 兼容性由 preflight 记录。

本地交付验证覆盖 CPU 逻辑与固定框架的静态接口。下列 GPU 命令尚未在目标服务器执行。环境文件是核查过直接依赖约束的候选组合，不应视为已完成 GPU 验证。

## 1. 克隆、安装与环境检查

```bash
git clone https://github.com/Vannico233/RL.git
cd RL
conda env create -f environment.yml
conda activate chart-rl
bash scripts/install_server.sh
python scripts/server_env_probe.py > runs/environment/server_env.json
python scripts/preflight.py
```

`preflight.json` 必须通过。它检查 Python、关键包版本、框架 commit、可见 GPU、BF16 矩阵运算及插件导入。它不加载 7B 模型，不代替后续 G1。

核心候选版本：Python 3.11、ms-swift 4.6.0、vLLM 0.11.2、PyTorch 2.9.0、Transformers 4.57.6。完整约束见 `constraints-server.txt`；实际环境保存在 `runs/environment/pip-freeze.txt`。不额外安装 FlashAttention，训练和评估使用 SDPA。

默认一张卡训练、一张卡提供 vLLM 生成服务。安装与下载需要访问 PyPI、GitHub 和 Hugging Face；如服务器使用代理，应配置服务器自己的网络设置，不复制开发电脑的地址。模型为公开资源，不需要在代码中写入账号密码。

### 安装时出现 Git `early EOF` / `unexpected eof`

旧版安装文件通过 Git 克隆 ms-swift；此错误表示源码传输中断，尚未进入依赖编译。PyPI 镜像不会接管 GitHub 的 Git 请求。更新项目后重新安装：

```bash
git pull --ff-only
conda activate chart-rl
bash scripts/install_server.sh
```

当前安装直接下载 GitHub 官方固定 commit 的约 16 MiB 源码压缩包，核查 SHA256，再由 pip 构建安装；不克隆历史。源码版本仍为 4.6.0 / `ad818a2cdfafaf4305ac40c4acd892a211978bc2`。preflight 同时支持旧的固定 Git 安装和新的哈希验证压缩包安装。

压缩包仍需要服务器能够访问 `codeload.github.com`；它解决克隆过程带来的额外传输，不保证所有网络问题消失。如果仍失败，保留新的下载错误继续检查网络或代理。不要为了这类传输中断关闭 SSL 校验，也不需要重建整个 Conda 环境。

若未来 GitHub 更改压缩格式，压缩包哈希可能变化；应重新核查源码后更新固定哈希，不直接跳过校验。

## 2. 下载固定模型与数据

```bash
python scripts/download_assets.py --output-root data
python scripts/prepare_run.py --assets data/assets.json --output data_manifests/v1
```

下载器固定模型和 ChartQA 的 revision，并记录数据压缩包哈希。准备脚本生成约 1,024 道训练题、128 道验证题、其中最多 50 道 pilot 题，以及完整去重后的官方测试题池。

原始数据的目录由 `data/assets.json` 中的 `dataset_root` 指定，名称可能带空格。后续命令应保留变量两侧的引号。

```bash
CHART_MODEL="$(python -c 'import json; print(json.load(open("data/assets.json"))["model_path"])')"
CHART_DATA_ROOT="$(python -c 'import json; print(json.load(open("data/assets.json"))["dataset_root"])')"
```

检查 `data_manifests/v1/audit.json` 中的移除数量、来源分布、最终规模与 manifest 哈希。若使用 `--test-limit`，必须在所有结果中注明固定测试子集。不要根据测试表现重选题目。

准备和导出目录均拒绝覆盖；更换数据规模或处理规则时使用新目录。模型、数据、导出图片及运行产物不会被 Git 提交。

## 3. G1：真实多轮更新验收

```bash
python scripts/train.py --prepared data_manifests/v1 --setting B2 \
  --output runs/B2-smoke --smoke
```

默认运行 8 个优化步骤，组大小 4。诊断题从训练题池选取，额外提示模型首先请求一个指定裁剪，以验证真实工具闭环；**该运行不参与效果对比，不证明模型学会了自主选区。**

启动器自动等待生成服务、启动训练、保存日志，并在结束或异常时停止自己创建的进程组。不会关闭其他 GPU 作业。可先加 `--dry-run` 查看命令。

主要输出：

| 文件 | 用途 |
| --- | --- |
| `preflight.json` | 本次服务器环境检查 |
| `launch-plan.json` | 实际命令、数据哈希、初始化模型与代码内容哈希 |
| `rollout.log`、`training.log` | 服务与训练日志 |
| `rollouts/*.json` | 每条轨迹的动作、图片尺寸、生成 token/log-prob 数量、adapter 注册状态 |
| `audit/training.jsonl` | 梯度、参数指纹、损失 token、图像网格、同步回执、峰值显存 |
| `audit/rewards.jsonl` | 按图表问题标识分组的真实奖励分布 |
| `g1-report.json` | 验收结果及未满足的检查项 |
| `checkpoint-*` | 框架保存的 adapter、优化器和训练状态 |

`g1-report.json` 的 `status` 必须为 `pass`，长训练才能启动。验收要求出现真实裁剪后的继续生成、非零有限梯度、参数变化、更新后的同步回执，以及后续带已注册 adapter 的生成。编码检查要求损失 token 与采样 token 对齐、动态图像网格一致、log-prob 数量匹配。

同步回执与后续 adapter 注册记录是运行证据，不是生成端全部 tensor 的逐字节校验。仍需查看一两条完整轨迹和框架日志。

如果奖励全部相同，不能宣称 GRPO 完成有效学习。可以使用新的输出目录，将诊断延长到 `--max-steps 16`，或检查训练题与答案格式。不要添加人为奖励差异来让检查通过。若格式、坐标或图片理解持续失败，先进行第 7 节的少量 SFT。

若安装、参数或服务启动失败，保留失败目录，修正后用新输出目录重试；只有已经产生完整训练 checkpoint 才使用恢复训练功能。

## 4. G2：先验证工具是否有价值

```bash
python scripts/evaluate_matrix.py --phase pilot --prepared data_manifests/v1 \
  --output runs/pilot
python scripts/render_cases.py --run runs/pilot/B0-O4
```

在相同 pilot 题上比较 B0 的 O0–O4。查看 `runs/pilot/results.csv` 和各条件下的 `cases/index.html`。如需人工有效区域诊断，准备每行 `id`、`regions` 的 JSONL，并使用下面的 D0 命令；坐标必须基于该题首张实际显示图的尺寸。

```bash
chart-rl evaluate --manifest data_manifests/v1/pilot.jsonl \
  --data-root "$CHART_DATA_ROOT" --model "$CHART_MODEL" \
  --policy D0 --diagnostic-regions /path/to/pilot-regions.jsonl \
  --output runs/pilot-D0
```

D0 文件必须覆盖所选验证题，不能含测试题。示例结构：

```json
{"id":"替换为验证题的真实id","regions":[[0,0,280,280]]}
```

区分识别、数值推理、选区、格式和评分错误。如果高分辨率整图已经更好且更省成本，或者正确局部区域也不改善答案，就记录这个结果，不通过压低整图分辨率制造优势。

## 5. B1/B2 训练与恢复

```bash
python scripts/train.py --prepared data_manifests/v1 --setting B1 \
  --output runs/B1 --gate-report runs/B2-smoke/g1-report.json

python scripts/train.py --prepared data_manifests/v1 --setting B2 \
  --output runs/B2 --gate-report runs/B2-smoke/g1-report.json
```

两组使用同一初始化、题池、评分规则与随机种子。默认 128 个优化步骤，组大小 4、微批次 1、梯度累积 8；这不表示已经遍历 1,024 道题。根据实测吞吐通过 `--max-steps` 调整预算，并从奖励记录统计实际题目覆盖。

每 64 步进行固定验证题池监控并保存 checkpoint；训练内验证每题采样一次。最终统一评估使用 Transformers 和贪心解码，不能把训练内采样奖励直接替代最终成绩。

恢复示例（使用实际存在的 checkpoint）：

```bash
python scripts/train.py --prepared data_manifests/v1 --setting B2 \
  --output runs/B2 --resume-from runs/B2/checkpoint-64 --max-steps 128 \
  --gate-report runs/B2-smoke/g1-report.json
```

恢复保留同一模型、数据、种子和代码内容，`--max-steps` 表示总步骤上限。不要只加载 adapter 后声称已经恢复优化器状态。

## 6. 选择 checkpoint 后固定测试协议

先在 `val.jsonl` 上比较候选 checkpoint。确认选择后，运行最终矩阵；下例中的 `checkpoint-128` 须替换为实际选定的 checkpoint。

```bash
python scripts/evaluate_matrix.py --phase final --prepared data_manifests/v1 \
  --b1-adapter runs/B1/checkpoint-128 --b2-adapter runs/B2/checkpoint-128 \
  --output runs/final
python scripts/render_cases.py --run runs/final/B2-O4
```

最终矩阵包含 B0/O0、B0/O4、B1/O0，以及固定 B2 checkpoint 下的 O0–O4。可用 `--resume` 保留同一配置下已经完整结束的条件。失败的单个条件需保留其日志并移到其他名称，再继续矩阵；不会凭目录存在就跳过失败运行。

单独进行验证或评估：

```bash
chart-rl evaluate --manifest data_manifests/v1/val.jsonl \
  --data-root "$CHART_DATA_ROOT" --model "$CHART_MODEL" \
  --adapter runs/B2/checkpoint-128 --policy O4 --output runs/B2-selected-val
```

配对比较（右侧减左侧，按整张图表重采样）：

```bash
chart-rl compare --left runs/final/B1-O0/scored.jsonl \
  --right runs/final/B2-O4/scored.jsonl --output runs/final/B2-minus-B1.json
chart-rl compare --left runs/final/B2-O2/scored.jsonl \
  --right runs/final/B2-O4/scored.jsonl --output runs/final/active-minus-fixed.json
```

置信区间反映固定 checkpoint 的评估样本不确定性，不代表多训练种子的稳定性。只做一个训练种子必须明确说明。

O2/O3 预先提供两个固定/随机区域，O4 自主决定是否使用最多两次工具。调用上限一致，实际图片数、生成长度和耗时不一定相同。`visual_tokens` 统计每次生成输入的视觉 token，包含重复历史图片；它不是 GPU FLOPs。评估采用相同 Transformers 后端，训练采样采用 vLLM，二者吞吐不能直接互比。

## 7. 可选的小规模工具 SFT

仅当协议不稳定时进行。人工核查少量**训练题**的区域，创建 JSONL：

```json
{"id":"训练manifest中的真实id","reference_size":[560,560],"regions":[[0,0,280,280]],"reviewed":true,"review_note":"说明该区域包含哪些与题目相关的可见证据"}
```

`reference_size` 必须与实际显示图一致；可以用零个区域表示直接作答示范。程序会使用真实训练答案，不生成虚构推理过程。

```bash
python scripts/prepare_sft.py --manifest data_manifests/v1/train.jsonl \
  --data-root "$CHART_DATA_ROOT" --reviews /path/to/reviewed-train-regions.jsonl \
  --output runs/sft-data --limit 128
python scripts/train_sft.py --model "$CHART_MODEL" \
  --dataset runs/sft-data/sft.jsonl --output runs/sft

# 替换为实际选定的 SFT checkpoint，合并成共同初始化。
CUDA_VISIBLE_DEVICES=0 swift export --adapters runs/sft/checkpoint-N \
  --merge_lora true --output_dir runs/common-init
```

随后 B0 评估、B1/B2 训练和新的 B2 smoke 均使用 `--initial-model runs/common-init`；单独 `chart-rl evaluate` 则用 `--model runs/common-init`。保留原始基座与 SFT 初始化的比较，避免将 SFT 收益全归因于 RL。

## 8. 整理交付证据

```bash
python scripts/summarize_training.py --run runs/B1
python scripts/summarize_training.py --run runs/B2
```

训练汇总按 request ID 去除重复编码，并区分训练与周期验证。视觉 token 记录覆盖已到达编码器的轨迹；更早失败的生成不会凭空估算。报告中的显存峰值仅属于训练进程，生成端需另行记录服务器监控。

填写 `reports/day1-feasibility.md` 和 `reports/experiment-report.md`。保留原始 manifest、来源分组、代码和模型哈希、运行计划、环境、日志、adapter、逐题输出、失败案例与成本覆盖率。

所有训练与科学结果以服务器实际产物为准。项目代码和本地测试完成，不等于主动观察优于基线，也不等于本轮实验已经完成。
