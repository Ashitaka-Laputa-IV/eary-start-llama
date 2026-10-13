"""交互式 TUI：常驻界面，方向键选模型、Enter 启停、状态实时刷新。

无参数运行 `eary-start-llama` 时进入这里；带子命令时仍走 cli.py 的一次性路径。

设计要点：
  - 启动模型会阻塞（等待就绪），所以放进 thread worker，界面不卡死
  - 状态每 2 秒重读一次，配合 state.py 的存活校验，外部杀进程也能反映出来
  - 日志屏按 1 秒增量读取，等价于 tail -f
"""

from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, RichLog

from . import config, engine, state


class LogScreen(Screen):
    """单个模型的日志页（增量跟随）。"""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "返回"),
        Binding("q", "app.pop_screen", "返回"),
        Binding("r", "reload", "重读"),
    ]

    def __init__(self, name: str, logfile: str) -> None:
        super().__init__()
        self.model_name = name
        self.logfile = logfile
        self._offset = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", highlight=False, markup=False, wrap=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"日志 · {self.model_name}"
        self.sub_title = self.logfile
        self._read_new()
        self.set_interval(1.0, self._read_new)

    def action_reload(self) -> None:
        self.query_one(RichLog).clear()
        self._offset = 0
        self._read_new()

    def _read_new(self) -> None:
        log = self.query_one(RichLog)
        try:
            with open(self.logfile, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                data = handle.read()
                self._offset = handle.tell()
        except OSError:
            return
        for line in data.splitlines():
            log.write(line)


class EslApp(App):
    """模型服务总览与操作界面。"""

    CSS = """
    DataTable { height: 1fr; }
    """
    TITLE = "eary-start-llama"
    BINDINGS = [
        Binding("s", "toggle", "启动 / 停止"),
        Binding("space", "toggle", "启动 / 停止", show=False),
        Binding("l", "logs", "日志"),
        Binding("r", "refresh", "刷新"),
        Binding("q", "quit", "退出"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []
        self._starting: set[str] = set()
        self._stopping: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="table", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Name", "Quant", "Size", "Status", "Endpoint", "PID")
        table.focus()
        self._reload()
        self.set_interval(2.0, self._reload)

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def _reload(self) -> None:
        table = self.query_one(DataTable)
        cursor = table.cursor_row

        table.clear()
        self._names = []
        models = config.load()
        runtimes = state.live()

        for name, model in models.items():
            self._names.append(name)
            runtime = runtimes.get(name)
            if name in self._starting:
                status = Text("● starting", style="bold yellow")
            elif name in self._stopping:
                status = Text("● stopping", style="bold yellow")
            elif runtime is not None:
                status = Text("● running", style="bold green")
            else:
                status = Text("● stopped", style="dim")

            quant, size = engine.describe(model)
            table.add_row(
                name,
                quant,
                size,
                status,
                runtime.endpoint if runtime else "-",
                str(runtime.pid) if runtime else "-",
            )

        if self._names:
            table.move_cursor(row=min(cursor, len(self._names) - 1))
        self.sub_title = f"{len(self._names)} models · {len(runtimes)} running"

    def _selected(self) -> str | None:
        if not self._names:
            return None
        row = self.query_one(DataTable).cursor_row
        return self._names[row] if 0 <= row < len(self._names) else None

    # ------------------------------------------------------------------ #
    # 操作
    # ------------------------------------------------------------------ #
    def action_refresh(self) -> None:
        self._reload()
        self.notify("已刷新")

    def action_toggle(self) -> None:
        name = self._selected()
        if name is None:
            return

        # 注意：start / stop 都会阻塞（分别等待进程就绪 / 退出），
        # 必须丢进后台线程，否则会冻住整个界面。
        if name in state.live():
            if name in self._stopping:
                return
            self._stopping.add(name)
            self._reload()
            self._stop_worker(name)
            return

        if name in self._starting:
            return
        self._starting.add(name)
        self._reload()
        self._start_worker(name)

    @work(thread=True, exclusive=False)
    def _start_worker(self, name: str) -> None:
        model = config.load().get(name)
        try:
            if model is None:
                self.call_from_thread(self.notify, f"unknown model: {name}", severity="error")
                return
            runtime = engine.start(model)
            self.call_from_thread(self.notify, f"{name} 已启动 → {runtime.endpoint}")
        except engine.StartError as exc:
            self.call_from_thread(
                self.notify, f"{name} 启动失败：{exc}", severity="error"
            )
        finally:
            self._starting.discard(name)
            self.call_from_thread(self._reload)

    @work(thread=True, exclusive=False)
    def _stop_worker(self, name: str) -> None:
        try:
            stopped = engine.stop(name)
            message = f"已停止 {name}" if stopped else f"{name} 未在运行"
            self.call_from_thread(self.notify, message)
        finally:
            self._stopping.discard(name)
            self.call_from_thread(self._reload)

    def action_logs(self) -> None:
        name = self._selected()
        if name is None:
            return
        runtime = state.live().get(name)
        logfile = runtime.logfile if runtime else str(config.LOG_DIR / f"{name}.log")
        self.push_screen(LogScreen(name, logfile))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable 把 Enter 绑定给了 select_cursor，这里接住它做启停
        self.action_toggle()


def run() -> None:
    EslApp().run()
