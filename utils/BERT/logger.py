"""実験ログを保存する

    * コンソール出力とは別で実験ログを保存しておけると便利だなと思って導入しました
    * 実装は MMBT の logger をほぼそのまま使っています
    * https://github.com/facebookresearch/mmbt/blob/master/mmbt/utils/logger.py
"""
from __future__ import annotations
import logging
import time
import datetime
from datetime import timedelta
from typing import Optional
import traceback

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)

class Log():
    _instance: Optional[Log] = None

    @staticmethod
    def get_instance() -> Log:
        if not Log._instance:
            Log._instance = Log()
        return Log._instance
    
    def __init__(self) -> None:
        self.created_dt = datetime.datetime.now()
        self._created_str = self.created_dt.strftime("%Y%m%d-%H%M%S")
        logfile = 'log/' + self._created_str + ".log"

        log_formatter = LogFormatter()

        file_handler = logging.FileHandler(logfile, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(log_formatter)

        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # reset logger elapsed time
        def reset_time():
            log_formatter.start_time = time.time()

        self.logger.reset_time = reset_time
    
    def set_args(self,args) -> None:
        self.logger.info(
            "\n".join(
                "%s: %s" % (k, str(v))
                for k, v in sorted(dict(vars(args)).items(), key=lambda x: x[0])
            )
        )

    def end_point(self, args, acc) -> None:
        elapsed_seconds = round(time.time() - self.logger.handlers[0].formatter.start_time)
        prefix = "プログラム終了 %s - 実行時間 %s\n" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        with open('summary.txt', 'a') as f:
            f.write(f"Log ID : {self.created_str}\n")
            f.write(prefix)
            f.write(args)
            f.write(f"\n正解率 : {acc:.4f}\n\n")

    @property
    def created_str(self) -> str:
        return self._created_str

    def debug(self, message) -> None:
        self.logger.debug(message)

    def info(self, message) -> None:
        self.logger.info(message)

    def cleanup(self) -> None:
        error = traceback.format_exc()
        if not error.endswith("SystemExit: 0\n"):
            self.logger.error(traceback.format_exc())