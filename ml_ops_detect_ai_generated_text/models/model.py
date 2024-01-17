import torch
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import roc_curve, auc


class TextClassificationModel(pl.LightningModule):
    def __init__(self, model_name=None, num_labels=None, learning_rate=None) -> None:
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
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name) #
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

    def predict(self, input_ids, attention_mask):
        outputs = self(input_ids, attention_mask)
        return torch.argmax(outputs.logits, dim=1)

    def prediction_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        logits = outputs.logits
        # Accuracy
        acc = self.accuracy(outputs.logits, labels)
        # AUC
        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        # Get probabilities for the positive class
        pos_probabilities = probabilities[:, 1].detach().cpu().numpy()
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), pos_probabilities)
        # Calculate AUC
        auc_score = auc(fpr, tpr) 
        
        return acc, auc_score

