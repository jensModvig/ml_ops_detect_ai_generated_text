import torch
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer



class TextClassificationModel(pl.LightningModule):
    def __init__(self, model_name, num_labels):
        super(TextClassificationModel, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=2e-5)



