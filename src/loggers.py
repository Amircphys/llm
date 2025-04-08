import logging
import logging.config

log_conf = {
    "version": 1,
    "formatters": {
        "simple": {"format": "%(asctime)s\t%(levelname)s\t%(message)s"},
        "processed": {"format": "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"}
    },
    "handlers": {
        "strim_handler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "processed",
        }
    },
    "loggers": {
        "strim_logger": {
            "level": "DEBUG",
            "handlers": ["strim_handler"],
        }
    }
}

logging.config.dictConfig(log_conf)
strim_logger = logging.getLogger("strim_logger")
file_logger = logging.getLogger("file_logger")
