import torch 
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
from innofw.core.models.torch.architectures.image_to_text.nic import NeuralImageCaption
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

class ImageToTextLightningModule(BaseLightningModule):
    """Lightning module for image to text architectures.
    
    Lightining module for training, evaluation and testing of image to text architectures.
    Provides a capability to train with teacher forcing.
    Evaluates the model using BLEU and ROUGE metrics.
    """

    def __init__(self, 
                model: NeuralImageCaption,
                losses,
                optimizer_cfg,
                scheduler_cfg, 
                max_caption_length: int = 128,
                *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.max_caption_length = max_caption_length
        self.losses = losses

    def log_losses(
        self, name: str, logits: torch.FloatTensor, truths: torch.LongTensor,
        on_step: bool = None, on_epoch: bool = None
    ) -> torch.FloatTensor:
        """Function to compute and log losses"""
        total_loss = 0
        for loss_name, weight, loss in self.losses:
            # for loss_name in loss_dict:
            local_loss = loss(logits, truths)
            total_loss += weight * local_loss

            self.log(
                f"loss/{name}/{weight} * {loss_name}",
                local_loss,
                on_step=on_step,
                on_epoch=on_epoch,
            
            )
    
        # val_loss and train_loss
        self.log(f"{name}_loss", total_loss, on_step=on_step, on_epoch=on_step)
        return total_loss
 
    def training_step(self, batch, batch_id):
        """Training step for image to text architectures.
        
        Args:
            batch (tuple): Batch of images, captions and lengths.
            batch_id (int): Batch id.

        Returns:
            torch.FloatTensor: Loss value.
        """
        images, captions, lengths = batch

        output = self.model.forward(images, captions, True)

        output_ = torch.nn.utils.rnn.pack_padded_sequence(
            output,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        captions_ = torch.nn.utils.rnn.pack_padded_sequence(
            captions,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        loss = self.log_losses("train",  output_.data, captions_.data)
        return loss
    
    def validation_step(self, batch, batch_id):
        """Validation step for image to text architectures.

        Performs validation step for image to text architectures. 
        Evaluates the model using BLEU and ROUGE metrics.
        
        Args:
            batch (tuple): Batch of images, captions and lengths.
            batch_id (int): Batch id.

        Returns:
            torch.FloatTensor: Loss value.
        """
        
        images, captions, lengths = batch
        output = self.model.forward(images)

        output_ = torch.nn.utils.rnn.pack_padded_sequence(
            output,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        captions_ = torch.nn.utils.rnn.pack_padded_sequence(
            captions,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        loss = self.log_losses("val", output_.data, captions_.data)

        text_captions = self.trainer.datamodule.tokenizer_model.Decode(captions.tolist())
        text_outputs = self.trainer.datamodule.tokenizer_model.Decode(output.argmax(dim=-1).tolist())

        
        # Measure BLEU-1, BLEU-2 and BLUE-4 score
        bleu1 = bleu_score(text_outputs, text_captions, n_gram=1)
        bleu2 = bleu_score(text_outputs, text_captions, n_gram=2)
        bleu4 = bleu_score(text_outputs, text_captions, n_gram=4)

        # Measure ROUGE-L score
        rouge_l = rouge_score(text_outputs, text_captions, rouge_keys=("rougeL"))

        # Measure METEOR score
        # meteor = meteor_score(text_outputs, text_captions)

        self.log("val_loss", loss)
        self.log("bleu1", bleu1)
        self.log("bleu2", bleu2)
        self.log("bleu4", bleu4)
        self.log("rouge_l", rouge_l)
        # self.log("meteor", meteor)
        
        return loss
    
    def test_step(self, batch, batch_id):
        """Test step for image to text architectures.

        Performs test step for image to text architectures. 
        Evaluates the model using BLEU and ROUGE metrics.
        
        Args:
            batch (tuple): Batch of images, captions and lengths.
            batch_id (int): Batch id.

        Returns:
            torch.FloatTensor: Loss value.
        """
        images, captions, lengths = batch
        output = self.model.forward(images)

        output_ = torch.nn.utils.rnn.pack_padded_sequence(
            output,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        captions_ = torch.nn.utils.rnn.pack_padded_sequence(
            captions,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        loss = self.log_losses("val", output_.data, captions_.data)
        loss = torch.nn.functional.cross_entropy(output, captions)

        text_captions = self.trainer.datamodule.tokenizer_model.Decode(captions.tolist())
        text_outputs = self.trainer.datamodule.tokenizer_model.Decode(output.argmax(dim=-1).tolist())

        
        # Measure BLEU-1, BLEU-2 and BLUE-4 score
        bleu1 = bleu_score(text_outputs, text_captions, n_gram=1)
        bleu2 = bleu_score(text_outputs, text_captions, n_gram=2)
        bleu4 = bleu_score(text_outputs, text_captions, n_gram=4)

        # Measure ROUGE-L score
        rouge_l = rouge_score(text_outputs, text_captions, rouge_keys=("rougeL"))

        # Measure METEOR score
        # meteor = meteor_score(text_outputs, text_captions)

        self.log("val_loss", loss)
        self.log("bleu1", bleu1)
        self.log("bleu2", bleu2)
        self.log("bleu4", bleu4)
        self.log("rouge_l", rouge_l)
        # self.log("meteor", meteor)
        
        return loss
    
    def predict_step(self, batch, batch_ids):
        """Predict step for image to text architectures.

        Only returns the predicted text captions.

        Args:
            batch (tuple): Batch of images and captions.
            batch_ids (int): Batch id.

        Returns:
            list: List of predicted text captions.
        """

        images, captions = batch
        output = self.model.forward(images)
        output = output.permute(0, 2, 1)

        text_outputs = self.trainer.datamodule.tokenizer_model.Decode(output.argmax(dim=-1).tolist())
        return text_outputs
    
