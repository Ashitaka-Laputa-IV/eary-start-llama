# eary-start-llama

用 `py-cli` + `json` 管理本机 [llama.cpp](https://github.com/ggml-org/llama.cpp) 模型服务：一份清单 + 一个交互界面，把 `llama serve` 的启动、停止、日志变成顺手操作。

它不参与推理，只是 `llama serve` 之上的一层编排。

## 安装

需要 Python >= 3.13、[uv](https://github.com/astral-sh/uv)，以及带 `serve` 的 `llama` 命令。

```bash
git clone <this-repo> && cd eary-start-llama
uv sync
```

## 使用

```bash
uv run eary-start-llama        # 无参数 → 进入交互界面
```

| 按键 | 作用 |
| --- | --- |
| ↑ / ↓ | 选择模型 |
| Enter / `s` | 启动 / 停止选中的模型 |
| `l` | 查看日志（`Esc` 返回） |
| `r` / `q` | 刷新 / 退出 |

也可以直接用子命令（适合脚本）：

```bash
uv run eary-start-llama list
uv run eary-start-llama start bge-m3
uv run eary-start-llama stop --all
uv run eary-start-llama log bge-m3 -f
```

可用命令：`list` `status` `start` `stop` `log` `add` `rm` `init`，详见 `uv run eary-start-llama -h`。

## 配置

声明要管哪些模型，写在项目根 `models.json`（完整示例见 [`example.model.json`](./example.model.json)）：

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

- `source`（必填）：HF 仓库 `user/model[:quant]`，或本地 `.gguf` 路径
- `args`（可选）：追加给 `llama serve` 的参数，如 `--embeddings`、`--rerank`、`-c 8192`
- `port`（可选）：固定端口；省略则自动分配

首次运行会自动从 `llama -cl` 导入已有缓存模型。

## 其他

- 运行状态与日志放在 `~/.local/state/eary-start-llama/`，不写进项目
- 清单查找顺序：`$ESL_CONFIG` > 项目根 `models.json` > `~/.config/eary-start-llama/models.json`
