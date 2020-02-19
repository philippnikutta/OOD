from ood.log import logger

#batch_logger =logging.getLogger('batch')

class BatchLogger(object):
    def __init__(self, total_epochs, total_batches):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
      
    def log(self, current_epoch, current_batch, loss, accuracy):
        logger.info('Epoch: {:<2}/{:>2}    [Batch: {:>5}/{:>6.0f}]  Loss: {:>2.4f}    Accuracy: {:>2.2f}'.format(
                            current_epoch, self.total_epochs, current_batch, self.total_batches, loss, accuracy                                 
                                ))
        
