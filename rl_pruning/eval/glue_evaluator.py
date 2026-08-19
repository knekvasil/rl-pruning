import torch
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class GLUEEvaluator:
    def __init__(
        self, model, tokenizer, task_name="sst2", device="cuda", batch_size=32
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        # Load validation set for faster eval
        self.dataset = datasets.load_dataset("glue", task_name)["validation"]

        def tokenize(batch):
            return tokenizer(
                batch["sentence" if task_name == "sst2" else "text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

        self.dataset = self.dataset.map(tokenize, batched=True)
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)

    def __call__(self, model):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("label")  # Remove from batch
                outputs = model(**batch, labels=labels)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def evaluate_subset(self, model, indices, batch_size=32):
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                rows = [self.dataset[i] for i in chunk]
                batch = {
                    k: torch.stack([r[k] for r in rows]).to(self.device)
                    for k in ["input_ids", "attention_mask", "label"]
                }
                labels = batch.pop("label")
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total
