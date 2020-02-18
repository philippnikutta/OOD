from ood.log import logger

#batch_logger =logging.getLogger('batch')

class batch_logger(object):
    def __init__(self, current_epoch):
        self.current_epoch=current_epoch
        
