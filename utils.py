import logging
logger = logging.getLogger(__name__)


def debug(*args):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(' '.join((str(arg) for arg in args)))

def info(*args):
    if logger.isEnabledFor(logging.INFO):
        logger.info(' '.join((str(arg) for arg in args)))
