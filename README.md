# eary-start-llama

用 `py-cli` + `json` 管理本机 [llama.cpp](https://github.com/ggml-org/llama.cpp) 模型服务。

一句话：**把散在 HF 缓存里的一堆 gguf，变成一份带状态的服务清单**，`list` / `start` / `stop` / `log` 四个动作搞定，输出用 [rich](https://github.com/Textualize/rich) 美化。

它本身不参与推理，只是 `llama serve` 之上薄薄的一层编排（wrapper）：记录"有哪些模型、跑没跑、在哪个端口、日志在哪"。

## 特性

- **清单化**：用 `models.json` 声明要管哪些模型、每模型的参数与端口
- **真进程管理**：直接拉起 `llama serve` 子进程，停止时先 `SIGTERM` 再 `SIGKILL`
- **状态自愈**：`state.json` 配合 PID 存活校验，进程意外退出会自动标记为 `stopped`
- **开箱即用**：首次运行自动从 `llama -cl` 导入现有缓存模型
- **好看**：rich 表格、状态圈、启动面板、实时日志

## 环境要求

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv)
- llama.cpp 新版 CLI（提供 `llama` 与 `llama serve` 命令），可用 `llama --version` 验证

## 安装

```bash
git clone <this-repo> && cd eary-start-llama
uv sync
```

## 快速开始

```bash
uv run eary-start-llama list            # 首次运行会自动导入缓存里的模型
uv run eary-start-llama start bge-m3    # 启动，等待就绪后显示端点
uv run eary-start-llama status          # 查看 PID / 端口 / 运行时长
uv run eary-start-llama log bge-m3 -f   # 实时跟随日志，Ctrl-C 退出
uv run eary-start-llama stop --all      # 停止全部
```

安装为命令后可以省略 `uv run`：

```bash
eary-start-llama            # 不带参数 → 进入交互界面
eary-start-llama list
```

## 交互界面

不带任何参数直接运行，会进入一个**常驻的 TUI**（而不是跑完就退），一览所有模型状态并直接操作：

```bash
uv run eary-start-llama
```

| 按键 | 作用 |
| --- | --- |
| ↑ / ↓ | 选择模型 |
| Enter / `s` / 空格 | 启动 / 停止选中的模型 |
| `l` | 查看该模型日志（`Esc` 返回） |
| `r` | 手动刷新 |
| `q` | 退出 |

- 状态每 2 秒自动刷新；启动中的模型显示 `● starting`，就绪后转为 `● running` 并显示端点与 PID
- 启动动作在后台线程执行，界面不会卡住
- 非交互环境（管道、重定向）下运行会自动退化为一次性列表，方便脚本使用

## 命令

| 命令 | 说明 |
| --- | --- |
| `list`（别名 `ls`） | 列出模型台账（含本地文件体积与运行状态） |
| `status [name]`（别名 `st`） | 查看运行状态：PID / 端口 / 运行时长 / 日志 |
| `start <name> [--port N] [--timeout S]` | 启动模型服务，等待端口就绪 |
| `stop <name> \| --all` | 停止服务（先 SIGTERM，超时再 SIGKILL） |
| `log <name> [-f]` | 查看日志，`-f` 实时跟随 |
| `add <source> [--name] [--args] [--port]` | 把一个模型加入清单 |
| `rm <name>` | 从清单移除（**不删除**模型文件） |
| `init [--force]` | 从 `llama -cl` 重新导入清单 |

以上子命令适合脚本化调用；日常操作直接用不带参数的交互界面更顺手（见上一节）。

## 配置：models.json

清单是"你手写的我想要什么"，也是唯一需要你维护的文件。

```json
{
  "models": {
    "bge-m3": {
      "source": "ggml-org/bge-m3-Q8_0-GGUF:Q8_0",
      "args": "--embeddings",
      "port": 9931
    }
  }
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| 键（如 `bge-m3`） | ✅ | 短名，命令行里使用 |
| `source` | ✅ | HF 仓库 `user/model[:quant]`，或本地 `.gguf` 绝对路径 |
| `args` | ❌ | 额外拼到 `llama serve` 后的参数，如 `--embeddings`、`--rerank`、`-c 8192` |
| `port` | ❌ | 固定端口；省略则从 9931 起自动找空位 |

- 完整示例见 [`example.model.json`](./example.model.json)（含嵌入 / 重排 / 聊天 / 本地文件四种写法）
- **清单查找顺序**：`$ESL_CONFIG` > `<项目根>/models.json` > `~/.config/eary-start-llama/models.json`
- 注意：以 `//` 开头的键只是注释，程序读取时忽略；但 `add` / `rm` 改写清单时会被丢弃

## 文件位置

| 内容 | 位置 | 进 git？ |
| --- | --- | --- |
| 模型清单 | `<项目根>/models.json` | ✅ 建议提交 |
| 运行状态 | `~/.local/state/eary-start-llama/state.json` | ❌ 运行时产物 |
| 日志 | `~/.local/state/eary-start-llama/logs/*.log` | ❌ 运行时产物 |

运行状态与日志刻意不放进项目：它们是本机的运行时数据，随机器而变，混进仓库只会造成噪音。

## 架构

设计上刻意把"代码"与"数据"分层，以便单独替换：

```
src/eary_start_llama/
  config.py   配置层：models.json 的读写与路径查找
  state.py    状态层：state.json + PID 存活校验（脏记录自动清理）
  engine.py   进程层：启动/停止 llama serve、探活、日志、来源解析
  ui.py       渲染层：rich 表格与面板
  cli.py      入口：argparse 命令分发
```

三层数据的职责边界：

- **配置**（你手写）：要管哪些模型
- **状态**（程序写）：现在跑没跑
- **缓存**（只读）：磁盘上到底有什么 —— 由 HF 缓存 / `llama -cl` 决定，不进清单

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `ESL_CONFIG` | 指定清单文件路径 |
| `ESL_STATE_DIR` | 指定状态与日志目录 |
| `ESL_LLAMA_BIN` | 指定 `llama` 可执行文件路径 |
| `HF_HUB_CACHE` | 指定 Hugging Face 缓存目录 |

## 常见问题

- **首次 `list` 是空的？** 需要 `llama` 在 PATH 中；也可用 `eary-start-llama init` 手动导入。
- **`source` 找不到本地文件？** 会在 `start` 时触发下载；可先用 `llama download -hf <repo>` 预下载。
- **端口被占用？** 不写 `port` 即自动分配；写了则启动时会自动避让已占用端口。
- **进程被外部杀掉，状态没更新？** 不会 —— `state.json` 每次读取都会做存活校验并自动清理。
