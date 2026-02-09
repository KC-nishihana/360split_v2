"""
ロギングユーティリティ - 360Split用
シンプルなログ設定とメッセージ出力
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


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


def setup_logger(name='360split', level=logging.INFO,
                 log_file=None, use_color=True):
    """
    ロガーをセットアップ

    Parameters:
    -----------
    name : str
        ロガー名
    level : int
        ログレベル（logging.DEBUG, INFO, WARNING等）
    log_file : str or Path, optional
        ログファイルパス。Noneの場合はファイル出力なし
    use_color : bool
        コンソール出力でカラーを使用するか

    Returns:
    --------
    logging.Logger
        設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # フォーマット定義
    if use_color:
        formatter = ColoredFormatter(
            fmt='%(levelname)s | %(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(levelname)s | %(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # ストリームハンドラ（コンソール）
    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # ファイルハンドラ
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_formatter = logging.Formatter(
            fmt='%(levelname)s | %(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name='360split'):
    """
    既に設定されたロガーを取得

    Parameters:
    -----------
    name : str
        ロガー名

    Returns:
    --------
    logging.Logger
        ロガーインスタンス
    """
    return logging.getLogger(name)


# デフォルトロガーを作成
default_logger = setup_logger('360split')
