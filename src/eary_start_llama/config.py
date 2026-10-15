"""配置层：模型清单（models.json）。

这份文件是"你手写的我想要什么"：
  - 要管哪些模型
  - 每个模型用哪个来源、哪些额外参数、偏好端口

它**不**记录"现在跑没跑"（那是 state.py 的事），也**不**记录
"磁盘上碰巧有什么"（那是 HF 缓存 / llama 的事）。

清单文件的查找顺序（先命中者生效）：
  1. $ESL_CONFIG                                显式指定路径，优先级最高
  2. <项目根>/models.json                        随项目走，便于 git 版本化
  3. ~/.config/eary-start-llama/models.json      全局兜底
若都不存在，则在项目根新建。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# src/eary_start_llama/config.py -> 上两级即项目根（源码运行时）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG = PROJECT_ROOT / "models.json"
GLOBAL_CONFIG = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "eary-start-llama"
    / "models.json"
)

# 是否运行在源码树中（开发模式）。pip / uv 安装进 site-packages 后，
# PROJECT_ROOT 会落到虚拟环境内部，不能再用它。
_IN_SOURCE_TREE = (PROJECT_ROOT / "pyproject.toml").is_file()


def _resolve_config() -> Path:
    env = os.environ.get("ESL_CONFIG")
    if env:
        return Path(env).expanduser()
    if PROJECT_CONFIG.exists():
        return PROJECT_CONFIG
    if GLOBAL_CONFIG.exists():
        return GLOBAL_CONFIG
    # 都不存在：源码开发写项目根方便版本管理，安装版写全局配置目录
    return PROJECT_CONFIG if _IN_SOURCE_TREE else GLOBAL_CONFIG


CONFIG_FILE = _resolve_config()

# 运行状态与日志（程序自己写的）统一放这里
STATE_DIR = Path(
    os.environ.get(
        "ESL_STATE_DIR",
        str(
            Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
            / "eary-start-llama"
        ),
    )
)
STATE_FILE = STATE_DIR / "state.json"
LOG_DIR = STATE_DIR / "logs"


@dataclass
class Model:
    """清单里的一个模型条目。"""

    name: str
    source: str  # "user/model[:quant]"（HF 仓库）或本地 .gguf 路径
    args: str = ""  # 追加到 `llama serve` 的额外参数
    port: int | None = None  # 偏好端口；None = 自动分配


def load() -> dict[str, Model]:
    """读取清单；文件不存在时返回空字典。"""
    if not CONFIG_FILE.exists():
        return {}
    raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    models: dict[str, Model] = {}
    for name, item in raw.get("models", {}).items():
        models[name] = Model(
            name=name,
            source=item.get("source", ""),
            args=item.get("args", ""),
            port=item.get("port"),
        )
    return models


def save(models: dict[str, Model]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"models": {name: _dump(m) for name, m in models.items()}}
    CONFIG_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def add(model: Model) -> None:
    models = load()
    models[model.name] = model
    save(models)


def remove(name: str) -> bool:
    models = load()
    if name not in models:
        return False
    del models[name]
    save(models)
    return True


def _dump(m: Model) -> dict:
    item: dict = {"source": m.source}
    if m.args:
        item["args"] = m.args
    if m.port:
        item["port"] = m.port
    return item
