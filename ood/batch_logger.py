from ood.log import logger

#batch_logger =logging.getLogger('batch')

class BatchLogger(object):
    def __init__(self, total_epochs, total_batches, writer):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.writer = writer
      
    def log(self, current_epoch, current_batch, loss, accuracy):
        logger.info('Epoch: {:<2}/{:>2}    [Batch: {:>5}/{:>6.0f}]  Loss: {:>2.4f}    Accuracy: {:>2.2f}'.format(
                            current_epoch, self.total_epochs, current_batch, self.total_batches, loss, accuracy                                 
                                ))
    
    def log_epoch(self, current_epoch, current_batch, loss, accuracy):
        self.writer.add_scalar(tag='Loss/Validation', scalar_value=loss, global_step=current_epoch)
        self.writer.add_scalar(tag='Accuracy/Validation', scalar_value=accuracy, global_step=current_epoch)
        logger.info('Epoch: {:<2}/{:>2}    Validation Loss:{:>2.4f}    Validation Accuracy: {:>2.2f}'.format(
                            current_epoch, self.total_epochs,  loss, accuracy                                 
                                ))



        
