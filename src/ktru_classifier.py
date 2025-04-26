import logging
from datetime import datetime
from typing import Dict, List, Tuple, Iterator
from tqdm.auto import tqdm, trange
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

DATA_PATH = "../data/test_df_for_ktru_classifier.csv"
CABLE_CLASS_NAME = 'Провода и кабели электронные и электрические прочие'
MIN_CLASS_FREQ = 10
COLUMNS = ['product_name', 'ktru_code', 'ktru_name']
TEST_SIZE = 0.25
BATCH_SIZE = 32
NUM_EPOCHS = 10
CLIP = 1
EMB_MODEL_NAME="DeepPavlov/rubert-base-cased"
MODEL_SAVE_PATH = "../models/classificator_ktru_rubert_cased_deep_pavlov.pt"
TENSORBOARD_LOGS_DIR = "new_logs"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s: %(message)s'
)
logger = logging.getLogger("ktru_classifier")


#####################################################  dataset  ####################################################

def get_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    df = pd.read_csv(data_path)
    count_dict = df['labels'].value_counts().to_dict()
    df['class_freq'] = df.labels.map(count_dict)

    df_clean = df[(df.class_freq > MIN_CLASS_FREQ) &
                  (df.ktru_name != CABLE_CLASS_NAME)]
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean[COLUMNS]
    df_clean.index = range(df_clean.shape[0])
    uniq_ktru_names = df_clean.ktru_name.unique()
    ktru_name_label_map = {name: id for id, name in enumerate(uniq_ktru_names)}
    df_clean['label'] = df_clean.ktru_name.map(ktru_name_label_map)
    df_train, df_valid = train_test_split(df_clean, test_size=TEST_SIZE)
    df_train.index = range(df_train.shape[0])
    df_valid.index = range(df_valid.shape[0])
    return df_train, df_valid, ktru_name_label_map


class DataSet(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx]['product_name'], self.data.iloc[idx]['label']


#####################################################  models  ####################################################

class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str, device: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        
    def forward(self, texts: List[str]):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)

        # Извлечение последнего скрытого состояния
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 1. Усреднение по всем токенам (кроме служебных)
        embeddings = last_hidden_state[:, 1:-1, :].mean(dim=1)  # Игнорируем [CLS] и [SEP]

        # 2. Использование эмбеддинга токена [CLS]
        cls_embedding = last_hidden_state[:, 0, :]
        return embeddings
    

class Classificator(nn.Module):
    def __init__(self, emb_model_name: str, num_classes: int, device: str):
        super().__init__()
        self.emb_model = EmbeddingModel(emb_model_name, device)
        self.fc_out = nn.Linear(768, num_classes)
        
    def forward(self, texts: Iterator[str]):
        embeddings = self.emb_model(texts)
        out = self.fc_out(embeddings)
        return out
    
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        

#####################################################  train  ##############################################################

def eval_step(
    model,
    valid_dataloader,
    criterion,
    device):
    model.eval()
    valid_loss = 0
    true_predict = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc='Valid', leave=False):
            texts, labels = batch[0], batch[1].to(device)
            output = model(texts)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            _, pred = output.topk(k=1)
            true_predict += (labels == pred.squeeze(1)).sum().item()
            total += len(labels)
            
        valid_loss /= len(valid_dataloader)
        epoch_acc = round(true_predict/total, 3)
    return valid_loss, epoch_acc


def train_step(
    model,
    train_dataloader,
    criterion,
    optimizer,
    clip,
    global_step,
    writer,
    device):
    model.train()
    train_loss = 0
    true_predict = 0
    total = 0 
    for batch in tqdm(train_dataloader, desc='Train', leave=False):
        texts, labels = batch[0], batch[1].to(device)
        output = model(texts)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        _, pred = output.topk(k=1)
        true_predict += (labels == pred.squeeze(1)).sum().item()
        total += len(labels)
        writer.add_scalar("Training/loss", loss.item(), global_step)
        global_step += 1
        
    train_loss /= len(train_dataloader)
    epoch_acc = round(true_predict/total, 3)
    return train_loss, epoch_acc, global_step

def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    clip,
    num_epochs,
    global_step,
    writer,
    device,
    ):
    last_global_step = 0
    best_acc = float('-inf')
    for epoch in trange(num_epochs, desc="Epochs"):
        train_loss, train_acc, last_global_step = train_step(
            model,
            train_dataloader,
            criterion,
            optimizer,
            clip,
            last_global_step,
            writer,
            device
        )
        writer.add_scalar("Evaluation/train_loss", train_loss, epoch)
        writer.add_scalar("Evaluation/train_accuracy", train_acc, epoch)
        logger.info(f"train loss: {round(train_loss, 3)}, train_accuracy: {train_acc}")
        
        valid_loss, valid_acc = eval_step(
            model,
            valid_dataloader,
            criterion,
            device
        )
        if valid_acc > best_acc:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_acc = valid_acc
            
        writer.add_scalar("Evaluation/valid_loss", valid_loss, epoch)
        writer.add_scalar("Evaluation/valid_accuracy", valid_acc, epoch)
        logger.info(f"valid loss: {round(valid_loss, 3)}, valid_accuracy: {valid_acc}")
        
    return train_acc, valid_acc

#####################################################    ##############################################################


def main():
    start_time = datetime.now()
    logger.info(f"Get dataloaders...")
    df_train, df_valid, ktru_name_label_map = get_data(DATA_PATH)
    train_dataset = DataSet(df_train)
    valid_dataset = DataSet(df_valid)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
    logger.info(f"Number of batchs in train_dataloader: {len(train_dataloader)}, in valid_dataloader: {len(valid_dataloader)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device is: {device}")
    model = Classificator(emb_model_name=EMB_MODEL_NAME, num_classes=len(ktru_name_label_map), device=device)
    model.to(device)
    num_params = count_model_parameters(model)
    logger.info(f"Number of params for model: {num_params}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    writer=SummaryWriter(TENSORBOARD_LOGS_DIR)
    logger.info(f"Start train model...")
    train_accuracy, valid_accuracy = train_model(
        model,
        train_dataloader,
        valid_dataloader,
        criterion,
        optimizer,
        clip=CLIP,
        num_epochs=NUM_EPOCHS,
        global_step=0,
        writer=writer,
        device=device,
        )
    end_time = datetime.now()
    logger.info(f"Time execution: {end_time-start_time}, train_perplexity: {train_perplexity}, valid_perplexity: {valid_perplexity}")
    
    
if __name__ == "__main__":
    main()