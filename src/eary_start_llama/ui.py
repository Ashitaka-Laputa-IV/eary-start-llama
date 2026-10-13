"""rich 渲染层：只负责"怎么显示"。

上层把配置与运行状态合并成 `Row` 视图对象传进来，
所以这一层同时可用于真实实现与将来的任何调整。
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class Row:
    """一行展示视图：配置信息 + 运行状态合并后的结果。"""

    name: str
    quant: str = "-"
    size: str = "-"
    running: bool = False
    endpoint: str = "-"
    pid: int | None = None
    uptime: str = "-"
    logfile: str = ""


def _status(running: bool) -> Text:
    if running:
        return Text("● running", style="bold green")
    return Text("● stopped", style="dim")


def _short_path(path: str, width: int = 56) -> str:
    if len(path) <= width:
        return path
    parts = path.split("/")
    if len(parts) > 2:
        return f"{parts[0]}/…/{parts[-1]}"
    return path[: width - 1] + "…"


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def render_list(rows: list[Row]) -> None:
    table = Table(
        box=box.ROUNDED,
        header_style="bold",
        title="[bold]Models[/bold]",
        title_justify="left",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Quant")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Status")
    table.add_column("Endpoint", style="green", no_wrap=True)

    for r in rows:
        table.add_row(r.name, r.quant, r.size, _status(r.running), r.endpoint)

    console.print(table)
    running = sum(1 for r in rows if r.running)
    console.print(f"[dim]{len(rows)} models · {running} running[/dim]")


def render_status(rows: list[Row]) -> None:
    table = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold",
        title="[bold]Runtime[/bold]",
        title_justify="left",
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Status")
    table.add_column("PID", justify="right")
    table.add_column("Port", justify="right")
    table.add_column("Uptime", justify="right")
    table.add_column("Log", style="dim")

    for r in rows:
        port = r.endpoint.rsplit(":", 1)[-1] if r.running else "-"
        log = f"logs/{r.name}.log" if r.running else "-"
        table.add_row(
            r.name,
            _status(r.running),
            str(r.pid) if r.pid else "-",
            port,
            r.uptime,
            log,
        )

    console.print(table)


def render_started(rt, model) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column()
    grid.add_row("source", model.source)
    grid.add_row("endpoint", rt.endpoint)
    grid.add_row("pid", str(rt.pid))
    grid.add_row("log", rt.logfile)

    console.print(
        Panel(
            grid,
            title=f"[bold green]{model.name}[/bold green]",
            title_align="left",
            border_style="green",
        )
    )
    console.print("[bold green]✔ running[/bold green]")


def render_stopped(name: str) -> None:
    console.print(f"[red]■[/red] stopped [cyan]{name}[/cyan]")


def render_error(msg: str) -> None:
    console.print(
        Panel(
            Text(msg, style="red"),
            title="[bold red]error[/bold red]",
            title_align="left",
            border_style="red",
        )
    )


def render_failure(name: str, tail: str) -> None:
    console.print(
        Panel(
            Text(tail or "(无日志输出)", style="dim"),
            title=f"[bold red]{name} 启动失败[/bold red]",
            title_align="left",
            border_style="red",
        )
    )


def print_log_header(name: str, logfile: str) -> None:
    console.print(
        Panel(
            f"[bold]{name}[/bold]  [dim]{logfile}[/dim]",
            title="[bold]log[/bold]",
            title_align="left",
            border_style="dim",
        )
    )


def print_log_line(line: str) -> None:
    # 日志是原始文本：关闭 markup（否则 [xxx] 会被当标签吃掉）、关闭自动换行
    console.print(line, highlight=False, markup=False, soft_wrap=True)


def info(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")
