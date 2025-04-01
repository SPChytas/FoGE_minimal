import pytorch_lightning as pl
import torch 
import numpy as np 

from models.llama2 import ModelArgs, Transformer


from utils.metrics import SeqAccuracy


class LightningArgs:
    model_args = ModelArgs
    lr: float = 1e-2
    ckpt_name: str = "../llama/llama-2-7b/consolidated.00.pth"


class FockLLM(pl.LightningModule):
    def __init__(self, lightning_args):
        super().__init__()

        self.lightning_args = lightning_args
        
        self.model = None 
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.metric = SeqAccuracy()


    def configure_model(self):
        if self.model is not None:
            return 
        self.model = Transformer(self.lightning_args.model_args)
        ckpt = torch.load(self.lightning_args.ckpt_name)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.freeze_llm()


    def configure_optimizers(self):
        return torch.optim.SGD([ param for param in self.model.parameters() if param.requires_grad == True], lr=self.lightning_args.lr, weight_decay=1e-2)

    def on_train_epoch_start(self):
        self.model.freeze_llm()
        self.metric.reset()

    def on_validation_epoch_start(self):
        self.metric.reset()

    def on_test_epoch_start(self):
        self.metric.reset()



    def training_step(self, batch):

        symbols, input_tokens, output_tokens, seq_lens = batch

        out = self.model(input_tokens, symbols, seq_lens)
        preds = torch.argmax(out, dim=-1)
        preds[output_tokens == -1] = -1

        loss = self.criterion(out.view(-1, self.lightning_args.model_args.vocab_size), output_tokens.view(-1))
        self.metric.update(preds, output_tokens)

        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        self.log('train_accuracy', self.metric.result().item(), prog_bar=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch):

        symbols, input_tokens, output_tokens, seq_lens = batch

        out = self.model(input_tokens, symbols, seq_lens)
        preds = torch.argmax(out, dim=-1)
        preds[output_tokens == -1] = -1

        self.metric.update(preds, output_tokens)

        self.log('val_accuracy', self.metric.result().item(), prog_bar=True, sync_dist=True)


    def test_step(self, batch):

        symbols, input_tokens, output_tokens, seq_lens = batch

        out = self.model(input_tokens, symbols, seq_lens)
        preds = torch.argmax(out, dim=-1)
        preds[output_tokens == -1] = -1

        self.metric.update(preds, output_tokens)

        self.log('test_accuracy', self.metric.result().item(), prog_bar=True, sync_dist=True)
