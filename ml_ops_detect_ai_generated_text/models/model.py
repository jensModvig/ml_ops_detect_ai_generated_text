import torch
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class TextClassificationModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, learning_rate=2e-5):
        super(TextClassificationModel, self).__init__()
        self.model_name = model_name
        # Using a pretrained distilbert model, with a single linear layer on
        # top
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        # we only want to update the classifier, not the entire model
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        #
        # self.tokenizer = DistilBertTokenizer.from_pretrained(model_name) #
        # NOTE: used?
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs.logits, labels)
        acc = self.accuracy(outputs.logits, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs.logits, labels)
        acc = self.accuracy(outputs.logits, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean()
