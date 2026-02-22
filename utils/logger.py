"""
ロギングユーティリティ - 360Split用

アプリケーション全体で統一されたロギングを提供する。
ルートロガー '360split' の下に階層的な子ロガーを配置し、
コンソール（カラー）とファイル（ローテーション）への出力を制御する。

使い方:
    # 各モジュールの冒頭で
    from utils.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("詳細デバッグ情報")
    logger.info("通常の処理情報")
    logger.warning("注意が必要な状況")
    logger.error("エラーが発生")
"""

import logging
import logging.handlers
import sys
import threading
import re
from pathlib import Path
from datetime import datetime

# アプリケーションのルートロガー名
ROOT_LOGGER_NAME = '360split'

# ログファイルのデフォルトパス
DEFAULT_LOG_DIR = Path(__file__).parent.parent / 'logs'
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / '360split.log'

# ログローテーション設定
MAX_LOG_BYTES = 5 * 1024 * 1024   # 5MB
BACKUP_COUNT = 3                   # 3世代保持


class ColoredFormatter(logging.Formatter):
    """カラー出力をサポートするフォーマッター"""

    COLORS = {
        'DEBUG': '\033[36m',      # シアン
        'INFO': '\033[32m',       # 緑
        'WARNING': '\033[33m',    # 黄
        'ERROR': '\033[31m',      # 赤
        'CRITICAL': '\033[35m',   # マゼンタ
    }
    RESET = '\033[0m'

    def format(self, record):
        """ログレコードをフォーマット"""
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET)
        record.levelname = f"{color}{levelname}{self.RESET}"
        return super().format(record)


def _short_name(name: str) -> str:
    """モジュール名を短縮表示用に変換する。

    例: 'core.video_loader' -> 'core.video_loader'
         '__main__'         -> 'main'
    """
    if name == '__main__':
        return 'main'
    return name


class _ShortNameFormatter(logging.Formatter):
    """モジュール名を短縮表示するフォーマッター"""

    def format(self, record):
        record.shortname = _short_name(record.name)
        return super().format(record)


class _ColoredShortNameFormatter(ColoredFormatter):
    """カラー＋モジュール名短縮フォーマッター"""

    def format(self, record):
        record.shortname = _short_name(record.name)
        return super().format(record)


# ---- 初期化済みフラグ ----
_initialized = False
_gui_log_entries = []
_gui_log_lock = threading.Lock()
_gui_buffer_handler = None
_GUI_BUFFER_MAX = 10000
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _GuiBufferHandler(logging.Handler):
    """GUI表示向けにログをメモリバッファへ蓄積するハンドラ。"""

    def __init__(self, level=logging.INFO):
        super().__init__(level=level)

    def emit(self, record):
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
                "level": logging.getLevelName(record.levelno),
                "logger": record.name,
                "message": record.getMessage(),
            }
            with _gui_log_lock:
                _gui_log_entries.append(entry)
                if len(_gui_log_entries) > _GUI_BUFFER_MAX:
                    overflow = len(_gui_log_entries) - _GUI_BUFFER_MAX
                    del _gui_log_entries[:overflow]
        except Exception:
            # GUIバッファ書き込み失敗はメインロギングへ影響させない
            return


def setup_logger(name=ROOT_LOGGER_NAME, level=logging.INFO,
                 log_file=None, use_color=True, enable_file_log=True):
    """
    ルートロガーをセットアップ（アプリケーション起動時に1回だけ呼ぶ）

    Parameters:
    -----------
    name : str
        ルートロガー名（通常変更不要）
    level : int
        ログレベル（logging.DEBUG, INFO, WARNING等）
    log_file : str or Path, optional
        ログファイルパス。Noneの場合はデフォルトパスを使用
    use_color : bool
        コンソール出力でカラーを使用するか
    enable_file_log : bool
        ファイル出力を有効にするか

    Returns:
    --------
    logging.Logger
        設定済みルートロガー
    """
    global _initialized

    logger = logging.getLogger(name)

    # 既に初期化済みの場合はレベルだけ更新して返す
    if _initialized:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    logger.setLevel(level)
    # 親ロガーへの伝搬を防止（二重出力回避）
    logger.propagate = False

    # ---- コンソールハンドラ ----
    console_fmt = (
        '%(levelname)s | %(asctime)s | %(shortname)s | %(message)s'
    )
    datefmt = '%Y-%m-%d %H:%M:%S'

    if use_color:
        console_formatter = _ColoredShortNameFormatter(
            fmt=console_fmt, datefmt=datefmt
        )
    else:
        console_formatter = _ShortNameFormatter(
            fmt=console_fmt, datefmt=datefmt
        )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(console_formatter)
    logger.addHandler(stream_handler)

    # ---- ファイルハンドラ（ローテーション付き） ----
    if enable_file_log:
        log_path = Path(log_file) if log_file else DEFAULT_LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_fmt = '%(levelname)s | %(asctime)s | %(shortname)s | %(message)s'
        file_formatter = _ShortNameFormatter(
            fmt=file_fmt, datefmt=datefmt
        )

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding='utf-8',
        )
        file_handler.setLevel(logging.DEBUG)  # ファイルには全レベル出力
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # ---- GUIメモリバッファハンドラ ----
    global _gui_buffer_handler
    if _gui_buffer_handler is None:
        _gui_buffer_handler = _GuiBufferHandler(level=logging.INFO)
        logger.addHandler(_gui_buffer_handler)

    _initialized = True
    return logger


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """
    モジュール用ロガーを取得する。

    ルートロガー '360split' の子ロガーを返す。
    例: get_logger('core.video_loader')
        → logging.getLogger('360split.core.video_loader')

    各モジュールでは以下のように使用する:
        from utils.logger import get_logger
        logger = get_logger(__name__)

    Parameters:
    -----------
    name : str
        モジュール名（通常 __name__ を渡す）

    Returns:
    --------
    logging.Logger
        ルートロガーの子ロガー
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + '.'):
        # 既にルートロガー名が含まれている場合はそのまま
        return logging.getLogger(name)
    return logging.getLogger(f'{ROOT_LOGGER_NAME}.{name}')


def set_log_level(level: int) -> None:
    """
    アプリケーション全体のログレベルを動的に変更する。

    Parameters:
    -----------
    level : int
        logging.DEBUG, logging.INFO 等
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)
    for handler in root.handlers:
        # ファイルハンドラは常にDEBUG（全記録）のままにする
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            continue
        handler.setLevel(level)


def enable_gui_log_buffer(level: int = logging.INFO) -> None:
    """GUIログバッファを有効化する。setup_logger後なら重複登録しない。"""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    global _gui_buffer_handler
    if _gui_buffer_handler is None:
        _gui_buffer_handler = _GuiBufferHandler(level=level)
        root.addHandler(_gui_buffer_handler)
    else:
        _gui_buffer_handler.setLevel(level)


def get_gui_log_entries(from_index: int = 0):
    """
    GUIバッファのログを取得する。

    Returns:
    --------
    tuple(list, int)
      (entries[from_index:], next_index)
    """
    with _gui_log_lock:
        safe_index = max(0, min(int(from_index), len(_gui_log_entries)))
        return list(_gui_log_entries[safe_index:]), len(_gui_log_entries)


def read_log_tail(max_lines: int = 1000, log_file: Path = None):
    """ログファイル末尾を行単位で取得する。"""
    path = Path(log_file) if log_file else DEFAULT_LOG_FILE
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-max(1, int(max_lines)):]]
    except Exception:
        return []


def strip_ansi(text: str) -> str:
    """ANSIエスケープを除去する。"""
    return _ANSI_RE.sub("", text or "")
