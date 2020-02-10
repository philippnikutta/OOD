import logging

logger = logging.getLogger('par')
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)-19.19s %(levelname)-1.1s %(filename)s:%(lineno)s] %(message)s')

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def set_verbosity(verbose):
  if verbose == 0:
    logger.setLevel(logging.WARNING)
  elif verbose == 1:
    logger.setLevel(logging.INFO)
  else:
    logger.setLevel(logging.DEBUG)

    
def add_output(fname, remove=True):
  if remove:
    logger.handlers = [
      h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
  h = logging.FileHandler(fname)
  h.setFormatter(formatter)
  logger.addHandler(h)

