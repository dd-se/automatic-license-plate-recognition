import logging
import os

LOGLEVEL = getattr(logging, os.getenv("LOGLEVEL", "INFO"), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOGLEVEL)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOGLEVEL)

        formatter = logging.Formatter("[%(asctime)s] | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False
    return logger
