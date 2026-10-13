"""命令行入口与分发：解析参数 -> 调用配置/状态/进程层 -> 交给 ui 渲染。"""

from __future__ import annotations

import argparse
import sys

from . import config, engine, state, ui


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("eary-start-llama")
    except Exception:
        return "0.0.0"


# --------------------------------------------------------------------------- #
# 组装视图
# --------------------------------------------------------------------------- #
def _prepare() -> None:
    """首次运行时，从 `llama serve -cl` 导入一份初始清单。"""
    if config.CONFIG_FILE.exists():
        return
    models = engine.discover()
    config.save(models)
    ui.info(f"首次运行：已从 llama 缓存导入 {len(models)} 个模型 → {config.CONFIG_FILE}")


def _rows(only: str | None = None) -> list[ui.Row]:
    models = config.load()
    runtimes = state.live()
    rows: list[ui.Row] = []
    for name, model in models.items():
        if only and name != only:
            continue
        rt = runtimes.get(name)
        quant, size = engine.describe(model)
        rows.append(
            ui.Row(
                name=name,
                quant=quant,
                size=size,
                running=rt is not None,
                endpoint=rt.endpoint if rt else "-",
                pid=rt.pid if rt else None,
                uptime=rt.uptime() if rt else "-",
                logfile=rt.logfile if rt else str(config.LOG_DIR / f"{name}.log"),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
def cmd_list(args) -> None:
    ui.render_list(_rows())


def cmd_status(args) -> None:
    rows = _rows(only=args.name)
    if args.name and not rows:
        ui.render_error(f"unknown model: {args.name}")
        sys.exit(1)
    ui.render_status(rows)


def cmd_start(args) -> None:
    models = config.load()
    model = models.get(args.name)
    if model is None:
        ui.render_error(f"unknown model: {args.name}")
        sys.exit(1)

    running = state.live().get(args.name)
    if running is not None:
        ui.console.print(
            f"[yellow]![/yellow] {args.name} 已在运行 → {running.endpoint}"
        )
        return

    try:
        with ui.console.status(
            f"[cyan]launching llama serve for [bold]{args.name}[/bold]...",
            spinner="dots",
        ):
            rt = engine.start(model, port=args.port, timeout=args.timeout)
    except engine.StartError as exc:
        ui.render_error(str(exc))
        ui.render_failure(args.name, exc.tail)
        sys.exit(1)

    ui.render_started(rt, model)


def cmd_stop(args) -> None:
    if args.all:
        runtimes = state.live()
        if not runtimes:
            ui.info("没有正在运行的模型")
            return
        for name in list(runtimes):
            if engine.stop(name):
                ui.render_stopped(name)
        return

    if not args.name:
        ui.render_error("请指定模型名，或使用 --all")
        sys.exit(1)

    if not engine.stop(args.name):
        ui.info(f"{args.name} 未在运行")
        return
    ui.render_stopped(args.name)


def cmd_log(args) -> None:
    models = config.load()
    if args.name not in models:
        ui.render_error(f"unknown model: {args.name}")
        sys.exit(1)

    runtime = state.live().get(args.name)
    logfile = runtime.logfile if runtime else str(config.LOG_DIR / f"{args.name}.log")
    ui.print_log_header(args.name, logfile)

    if args.follow:
        try:
            for line in engine.follow_log(logfile):
                ui.print_log_line(line)
        except KeyboardInterrupt:
            ui.console.print("\n[dim]已取消跟随[/dim]")
        return

    lines = engine.read_log(logfile)
    if not lines:
        ui.info("暂无日志（该模型还没启动过）")
        return
    for line in lines:
        ui.print_log_line(line)


def cmd_add(args) -> None:
    models = config.load()
    name = args.name or engine.derive_name(args.source.split(":")[0])
    if name in models:
        ui.render_error(f"模型已存在：{name}（用 rm 先删，或换 --name）")
        sys.exit(1)

    config.add(config.Model(name=name, source=args.source, args=args.args, port=args.port))
    quant, size = engine.describe(config.Model(name=name, source=args.source))
    path = engine.resolve_source_path(args.source)
    if path is None:
        ui.info("提示：本地未见该模型文件，首次 start 时将触发下载")
    ui.console.print(
        f"[green]✔[/green] 已添加 [cyan]{name}[/cyan] "
        f"({quant}, {size}) → {config.CONFIG_FILE}"
    )


def cmd_rm(args) -> None:
    if not config.remove(args.name):
        ui.render_error(f"unknown model: {args.name}")
        sys.exit(1)
    ui.console.print(f"[red]■[/red] 已从清单移除 [cyan]{args.name}[/cyan]（未删除模型文件）")


def cmd_init(args) -> None:
    found = engine.discover()
    if args.force:
        merged = found
    else:
        merged = config.load()
        for name, model in found.items():
            merged.setdefault(name, model)
    config.save(merged)
    ui.console.print(
        f"[green]✔[/green] 已从 `llama -cl` 导入 "
        f"{len(found)} 个模型（清单现有 {len(merged)} 个）"
    )
    ui.render_list(_rows())


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eary-start-llama",
        description="用 py-cli + json 管理 llama 模型服务",
    )
    parser.add_argument(
        "--version", action="version", version=f"eary-start-llama {_version()}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("list", aliases=["ls"], help="列出模型台账")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("status", aliases=["st"], help="查看运行状态")
    p.add_argument("name", nargs="?", help="只查看某个模型")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("start", help="启动模型服务")
    p.add_argument("name", help="模型短名")
    p.add_argument("--port", type=int, help="指定端口（默认从 9931 起自动分配）")
    p.add_argument(
        "--timeout", type=float, default=180.0, help="等待就绪的秒数（默认 180）"
    )
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("stop", help="停止模型服务")
    p.add_argument("name", nargs="?", help="模型短名")
    p.add_argument("--all", action="store_true", help="停止全部运行中的模型")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("log", help="查看模型日志")
    p.add_argument("name", help="模型短名")
    p.add_argument("-f", "--follow", action="store_true", help="实时跟随")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("add", help="把一个模型加入清单")
    p.add_argument("source", help="HF 仓库 user/model[:quant] 或本地 .gguf 路径")
    p.add_argument("--name", help="短名（默认自动派生）")
    p.add_argument("--args", default="", help="附加给 llama serve 的参数")
    p.add_argument("--port", type=int, help="偏好端口")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("rm", help="从清单移除模型")
    p.add_argument("name", help="模型短名")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("init", help="从 llama 缓存重新导入清单")
    p.add_argument("--force", action="store_true", help="用发现结果覆盖现有清单")
    p.set_defaults(func=cmd_init)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _prepare()

    if not getattr(args, "command", None):
        # 交互终端进 TUI；被管道/重定向时退化为一次性列表
        if sys.stdout.isatty():
            from .tui import run as run_tui

            run_tui()
        else:
            ui.render_list(_rows())
        return

    args.func(args)
