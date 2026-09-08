import logging
import os


class CustomFormatter(logging.Formatter):
    grey = "\x1b[90m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    blue = "\x1b[34m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32m"
    reset = "\x1b[0m"
    format = "%(asctime)s [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: f"{grey}[%(asctime)s]{reset} {blue}[%(levelname)s]{reset} %(message)s",
        logging.INFO: f"{grey}[%(asctime)s]{reset} {green}[%(levelname)s]{reset} %(message)s",
        logging.WARNING: f"{grey}[%(asctime)s]{reset} {yellow}[%(levelname)s]{reset} %(message)s",
        logging.ERROR: f"{grey}[%(asctime)s]{reset} {red}[%(levelname)s]{reset} %(message)s",
        logging.CRITICAL: f"{grey}[%(asctime)s]{reset} {bold_red}[%(levelname)s]{reset} %(message)s",
    }

    def format(self, record):  # noqa: F811
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # keep output local to this logger

    # Only check handlers attached to *this* logger
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)  # optional; handler can be NOTSET too
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    return logger


# Example usage
if __name__ == "__main__":
    logger = get_logger("example_logger")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
