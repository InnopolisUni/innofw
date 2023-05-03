import torch 
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule

class ImageToTextLightningModule(BaseLightningModule):
    
    def __init__(self, 
                model,
                losses,
                optimizer_cfg,
                scheduler_cfg, 
                max_caption_length: int=128, 
                use_teacher_forcing=False,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.max_caption_length = max_caption_length
        self.use_teacher_forcing = use_teacher_forcing

        # Expects the first loss 
        # to be the cross entropy loss
        # Others are ignored
        self.losses = losses[0][2]
    
    def setup(self, stage: str):
        if self.trainer and self.trainer.datamodule and not self.model.is_ready:
            self.model.initialize(self.trainer.datamodule.word2int)

    def training_step(self, batch, batch_ids):
        captions, image = batch
        outputs = self.model(
            image,
            captions,
            max_caption_length=self.max_caption_length,
            teacher_forcing=self.use_teacher_forcing
        )
        outputs = outputs.permute(0, 2, 1)
        loss = self.losses(outputs, captions)
        return {"loss": loss}