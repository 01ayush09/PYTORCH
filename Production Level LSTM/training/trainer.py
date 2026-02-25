import torch
from tqdm import tqdm
from training.metrics import compute_metrics

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, lengths)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc, f1 = compute_metrics(all_preds, all_labels)
        return total_loss / len(dataloader), acc, f1

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, lengths)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc, f1 = compute_metrics(all_preds, all_labels)
        return total_loss / len(dataloader), acc, f1