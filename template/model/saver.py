import tensorflow as tf

class Saver():
    def __init__(self):
        self.old_loss = None
    
    def save_model_good_valloss(self, model, config, manager, epoch):
        new_loss = model.valid_loss.result()
        if self.old_loss is not None and self.old_loss >= new_loss:
            message = "model was saved!{:.4f} -> {:.4f}".format(self.old_loss, new_loss)
            self.old_loss = new_loss
            manager.save(tf.Variable(epoch))
        
        elif self.old_loss is None:
            message = "init saver loss."
            self.old_loss = new_loss
        
        else:
            message = "pass. Best Validation loss:{:.4f}".format(self.old_loss)
        
        return message