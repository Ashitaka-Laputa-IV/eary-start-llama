"""最小冒烟测试：保证包能导入、CLI 能解析、关键工具函数行为正常。

不依赖 llama 命令或真实模型，可在 CI 中稳定运行。
"""

from __future__ import annotations

import os

import eary_start_llama
from eary_start_llama import config, state
from eary_start_llama.cli import build_parser


def test_has_main() -> None:
    assert callable(eary_start_llama.main)


def test_parser_accepts_subcommands() -> None:
    parser = build_parser()
    assert parser.parse_args(["list"]).command == "list"
    assert parser.parse_args(["start", "bge-m3"]).name == "bge-m3"


def test_config_file_name() -> None:
    assert config.CONFIG_FILE.name == "models.json"


def test_is_alive() -> None:
    assert state.is_alive(os.getpid())
    assert not state.is_alive(0)
