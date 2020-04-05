from ood.log import logger


class BatchLogger(object):
    def __init__(self, total_epochs, total_batches, writer, interval=0):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.writer = writer
        self.interval = interval

    def log(self, current_epoch, current_batch, loss, accuracy, write_to_tensorboard=False):
        if accuracy:
            if write_to_tensorboard:
                self.writer.add_scalar(
                    tag='Loss/Training', scalar_value=loss, global_step=current_batch)
                self.writer.add_scalar(
                    tag='Accuracy/Training', scalar_value=accuracy, global_step=current_batch)
            if current_batch % self.interval == 0:
                logger.info('Epoch: {:<2}/{:>2}    [Batch: {:>5}/{:>6.0f}]  Loss: {:>2.3f}    Accuracy: {:>2.2f}'.format(
                    current_epoch, self.total_epochs, current_batch, self.total_batches, loss, accuracy
                ))

        else:
            if write_to_tensorboard:
                self.writer.add_scalar(
                    tag='Loss/Training', scalar_value=loss, global_step=current_batch)
            if current_batch % self.interval == 0:
                logger.info('Epoch: {:<2}/{:>2}    [Batch: {:>5}/{:>6.0f}]  Loss: {:>2.5f}'.format(
                    current_epoch, self.total_epochs, current_batch, self.total_batches, loss
                ))

    def log_ae(self, current_epoch, current_batch, loss, loss_kld, loss_l1, write_to_tensorboard=False):
        if write_to_tensorboard:
            self.writer.add_scalar(
                tag='Sum Loss/Training', scalar_value=loss, global_step=current_batch)
            self.writer.add_scalar(
                tag='KLD Loss/Training', scalar_value=loss_kld, global_step=current_batch)
            self.writer.add_scalar(
                tag='L1 Loss/Training', scalar_value=loss_l1, global_step=current_batch)
        if current_batch % self.interval == 0:
            logger.info('Epoch: {:<2}/{:>2}    [Batch: {:>5}/{:>6.0f}]  Loss Sum: {:>2.5f}  Loss KLD: {:>2.5f}  Loss L1: {:>2.5f}'.format(
                current_epoch, self.total_epochs, current_batch, self.total_batches, loss, loss_kld, loss_l1
            ))

    def log_epoch(self, current_epoch, loss, accuracy):
        self.writer.add_scalar(tag='Loss/Validation',
                               scalar_value=loss, global_step=current_epoch)
        self.writer.add_scalar(tag='Accuracy/Validation',
                               scalar_value=accuracy, global_step=current_epoch)
        logger.info('Epoch: {:<2}/{:>2}    Validation Loss:{:>2.3f}    Validation Accuracy: {:>2.3f}'.format(
            current_epoch, self.total_epochs,  loss, accuracy
        ))

    def log_epoch_ae(self, current_epoch, loss, loss_shifted):
        self.writer.add_scalar(tag='Loss/Validation',
                               scalar_value=loss, global_step=current_epoch)
        self.writer.add_scalar(tag='Loss/Validation Shifted',
                               scalar_value=loss_shifted, global_step=current_epoch)
        logger.info('Epoch: {:<2}/{:>2}    Validation Loss:{:>2.3f}    Validation Shifted Loss: {:>2.3f}'.format(
            current_epoch, self.total_epochs,  loss, loss_shifted
        ))
