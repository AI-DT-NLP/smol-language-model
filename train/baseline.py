import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import json

class CustomDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)

def train_model(data_path, model_path="./model/checkpoints/baseline", epochs=3, batch_size=8, lr=1e-4):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).cuda()
    model.train()

    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.cuda()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model(data_path="./datasets/preprocessed/baseline.json")
