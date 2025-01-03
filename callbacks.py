import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, Callback


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_start(self, trainer, pl_module):
        print("Starting to train")
    
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        
        val_imgs = self.val_imgs.to(device=pl_module.device)
        x = val_imgs.reshape(val_imgs.shape[0],-1)

        logits = pl_module(x)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, caption=f"Pred:{str(int(pred))}, Label:{str(int(y))}") 
                         for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })