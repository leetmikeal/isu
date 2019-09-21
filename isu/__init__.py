__version__ = '0.9.0'

__all__ = [
]


def get_module_logger(modname):
    import logging
    from src import settings
    LOG = settings.LOG

    logger = logging.getLogger(modname)

    if LOG == 'stdout' or LOG == '':
        handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
