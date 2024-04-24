import logging

FORMAT = "[%(name)s.%(funcName)s] %(asctime)s |%(levelname)s| %(message)s"
FORMATTER = logging.Formatter(FORMAT)


def get_logger(obj, level=logging.INFO):
    # simple logger
    handler = logging.StreamHandler()
    handler.setFormatter(FORMATTER)
    name = (f"{obj.__class__.__module__}."
            f"{obj.__class__.__qualname__}"
            f"({id(obj)})")
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = 0
    return logger