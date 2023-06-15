from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import pdb
from torch.utils.data import DataLoader

dataset = load_dataset("sst2")

tokenizer = T5Tokenizer.from_pretrained("t5-base")

def tokenize_dataset(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_dataset, batched=True)

model = T5ForConditionalGeneration.from_pretrained("t5-base", num_labels=2)


num_epochs = 3
batch_size = 8
learning_rate = 1e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        # turn list to tensor
        input_ids = torch.stack(batch["input_ids"], dim=0)
        attention_mask = torch.stack(batch["attention_mask"],dim=0)
        labels = torch.tensor(batch["label"]).unsqueeze(1)

        optimizer.zero_grad()


        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_loss}")

    # Evaluation
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            _, predicted_labels = torch.max(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Epoch {epoch+1}/{num_epochs} - Evaluation Accuracy: {accuracy}")

# Test
test_dataloader = DataLoader(tokenized_dataset["test"], batch_size=batch_size)
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        _, predicted_labels = torch.max(logits, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)

test_accuracy = total_correct / total_samples
print(f"Test Accuracy: {test_accuracy}")
