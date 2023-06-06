import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import csv
import random
# 分类数据集类
class TextDataset(Dataset):
    def __init__(self, ids, texts, labels, tokenizer):
        self.ids = ids
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        id = self.ids[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'id': id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.LongTensor([label])
        }

# 定义训练函数
def train_model(train_data, model, device, epochs=3, batch_size=32, lr=1e-5):
    model.to(device)
    model.train()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_values = []
    acc_values = []
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].squeeze().to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, predicted_labels = torch.max(logits, dim=1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

            # 每10个步骤打印损失和准确率
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / 10
                avg_acc = total_correct / total_samples

                loss_values.append(avg_loss)
                acc_values.append(avg_acc)

                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss}, Accuracy: {avg_acc}")
                total_loss = 0
                total_correct = 0
                total_samples = 0

    # 保存模型
    output_dir = "./model_finetuned/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

train_ids, train_texts, train_labels = [], [], []
f = open('./sdp_act/train.txt')
for line in f.readlines()[1:]:
    tokens = line.strip().split('\t')
    train_ids.append(tokens[0])
    train_texts.append(tokens[7].replace('#AUTHOR_TAG', '').lower())
    train_labels.append(int(tokens[-1]))

# ids, texts, labels = [], [], []
# for id, text, label in zip(train_ids, train_texts, train_labels):
#     if label == 5:
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
#     if label == 3:
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
#     if label == 4:
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
#         ids.append(id)
#         texts.append(text)
#         labels.append(label)
# train_ids.extend(ids)
# train_texts.extend(texts)
# train_labels.extend(labels)


# 加载BERT模型和tokenizer
model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(set(train_labels)))
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# 创建训练集的数据集实例
train_data = TextDataset(train_ids, train_texts, train_labels, tokenizer)

# 训练模型并绘制损失和准确率曲线
train_model(train_data, model, device='cuda', epochs=5)
# 加载BERT模型和tokenizer
# model = AutoModelForSequenceClassification.from_pretrained('./model_finetuned', num_labels=len(set(train_labels)))
# tokenizer = AutoTokenizer.from_pretrained('./model_finetuned')


# 定义预测函数
def predict(id, text, model, tokenizer, device):
    model.to(device)
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=32,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].squeeze().to(device)
    attention_mask = encoding['attention_mask'].squeeze().to(device)
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label

# 验证集效果
acc_cnt = 0
f = open('./sdp_act/test.txt')
cnt = 0
for line in f.readlines()[1:]:
    cnt += 1
    tokens = line.strip().split('\t')
    predicted_label = predict(tokens[0], tokens[7].replace('#AUTHOR_TAG', '').lower(), model, tokenizer, device='cuda')

    if predicted_label == int(tokens[-1]):
        acc_cnt += 1
print ('test data acc is ', acc_cnt / cnt)