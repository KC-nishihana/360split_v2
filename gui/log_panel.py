"""
解析ログパネル - 360Split GUI
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QPlainTextEdit, QFileDialog
)

from utils.logger import (
    enable_gui_log_buffer,
    get_gui_log_entries,
    read_log_tail,
    strip_ansi,
    DEFAULT_LOG_FILE,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class LogPanel(QWidget):
    """解析ログ表示パネル。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[Dict[str, str]] = []
        self._cursor = 0
        self._setup_ui()
        enable_gui_log_buffer(logging.INFO)
        self._load_initial_file_logs()
        _, self._cursor = get_gui_log_entries(0)
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll_live_logs)
        self._timer.start()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("フィルタ:"))
        self._filter = QComboBox()
        self._filter.addItems(["All", "Analysis", "Warning+"])
        self._filter.currentTextChanged.connect(self._render)
        row.addWidget(self._filter)

        row.addWidget(QLabel("検索:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("キーワード")
        self._search.textChanged.connect(self._render)
        row.addWidget(self._search, stretch=1)

        self._btn_clear = QPushButton("クリア")
        self._btn_clear.clicked.connect(self._clear_view)
        row.addWidget(self._btn_clear)

        self._btn_save = QPushButton("保存")
        self._btn_save.clicked.connect(self._save_view)
        row.addWidget(self._btn_save)

        layout.addLayout(row)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._text, stretch=1)

    @staticmethod
    def _parse_file_line(line: str) -> Dict[str, str]:
        clean = strip_ansi(line)
        parts = [p.strip() for p in clean.split("|", 3)]
        if len(parts) == 4:
            return {
                "timestamp": parts[1],
                "level": parts[0],
                "logger": parts[2],
                "message": parts[3],
            }
        return {
            "timestamp": "",
            "level": "INFO",
            "logger": "unknown",
            "message": clean,
        }

    def _load_initial_file_logs(self):
        for line in read_log_tail(max_lines=1000, log_file=Path(DEFAULT_LOG_FILE)):
            self._rows.append(self._parse_file_line(line))
        self._render()

    def _poll_live_logs(self):
        entries, next_cursor = get_gui_log_entries(self._cursor)
        self._cursor = next_cursor
        if not entries:
            return
        for e in entries:
            self._rows.append({
                "timestamp": str(e.get("timestamp", "")),
                "level": strip_ansi(str(e.get("level", "INFO"))),
                "logger": str(e.get("logger", "")),
                "message": str(e.get("message", "")),
            })
        if len(self._rows) > 5000:
            self._rows = self._rows[-5000:]
        self._render(auto_scroll=True)

    def _row_visible(self, row: Dict[str, str]) -> bool:
        mode = self._filter.currentText()
        level = row.get("level", "INFO").upper()
        msg = row.get("message", "")
        logger_name = row.get("logger", "")
        text = f"{row.get('timestamp', '')} {level} {logger_name} {msg}"
        kw = self._search.text().strip().lower()
        if kw and kw not in text.lower():
            return False
        if mode == "Warning+":
            return level in {"WARNING", "ERROR", "CRITICAL"}
        if mode == "Analysis":
            candidates = (
                "analysis_request", "analysis_result", "analysis_finished",
                "stage_start", "stage_summary", "worker_start", "worker_finished",
                "Stage 0", "Stage 1", "Stage 2", "Stage 3",
            )
            return any(token in msg or token in logger_name for token in candidates)
        return True

    def _render(self, _=None, auto_scroll: bool = False):
        lines = []
        for row in self._rows:
            if not self._row_visible(row):
                continue
            lines.append(
                f"{row.get('timestamp', '')} | {row.get('level', ''):<8} | "
                f"{row.get('logger', '')} | {row.get('message', '')}"
            )
        self._text.setPlainText("\n".join(lines))
        if auto_scroll:
            sb = self._text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _clear_view(self):
        self._rows.clear()
        self._text.clear()

    def _save_view(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "ログを保存",
            str(Path.home() / "360split_gui_logs.txt"),
            "Text (*.txt);;All files (*)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._text.toPlainText())
