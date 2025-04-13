
#####################################################        import           #########################################

from datetime import datetime
from typing import List, Tuple, Dict
import logging
from typing import List, Optional
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(levelname)s: %(message)s')
logger = logging.getLogger("cnn")

#####################################################       constants         ##########################################

FILE_PATH = "../data/reviews.csv"
LOG_DIR = "regression"
TEXT_COLUMNS = ["Title", "FullDescription"]
CATEGORICAL_COLUMNS  = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
TARGET_COLUMN = "SalaryNormalized"
LOG_TARGET_COLUMN = "Log1pSalary"
MIN_TOKEN_COUNT = 10
TEST_SIZE = 0.2 
RANDOM_STATE = 42
UNK = "UNK"
PAD = "PAD"
NUM_EPOCHS=10
BATCH_SIZE=256



#####################################################        dataset          ##############################################

class DataSet:
    def __init__(self, file_path: str=FILE_PATH):
        self.df = self.get_df(file_path)
        self.tokenizer = WordPunctTokenizer()
        self.vocab = self.get_vocab()
        self.categorical_vectorizer = self.get_categorical_vectorizer()
        self.train_iterator, self.valid_iterator = self.get_iterators()
    
        
    def get_vocab(self):
        token_counts = Counter()
        for column in TEXT_COLUMNS:
            for text in self.df[column].tolist():
                token_counts.update(text.split())
            
        tokens = [token for token, count_token in token_counts.items() if count_token>=MIN_TOKEN_COUNT]
        tokens = [UNK, PAD] + sorted(tokens)
        token_to_idx = {token: idx for idx, token in enumerate(tokens)}
        return token_to_idx
        
    @staticmethod
    def get_df(file_path):
        df = pd.read_csv(file_path, index_col=None)
        df[LOG_TARGET_COLUMN] = np.log1p(df[TARGET_COLUMN]).astype("float32")
        df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna("NaN")
        df[TEXT_COLUMNS] = df[TEXT_COLUMNS].fillna("NaN")
        return df
        
    def as_matrix(self, sequences: List[List[str]], max_len: Optional[int])-> np.array:
        """Convert a list of tokens into a matrix with padding"""
        max_seq_len = max([len(seq) for seq in sequences])
        max_len = min(max_seq_len, max_len) if max_len else max_seq_len
        matrix = np.full(shape = (len(sequences), max_seq_len), fill_value=self.vocab[PAD])
        for i, seq in enumerate(sequences):
            tokens = seq.split()
            for j, token in enumerate(tokens):
                if j >= max_len:
                    continue
                matrix[i][j] = self.vocab.get(token, self.vocab[UNK])
        return torch.tensor(matrix)
    
    def get_categorical_vectorizer(self)-> DictVectorizer:
        company_counts = Counter(self.df["Company"])
        top_companies = set(name for name, count in company_counts.most_common(1000))
        self.df["Company"] = self.df["Company"].apply(
            lambda company: company if company in top_companies else "Other"
        )

        categorical_vectorizer = DictVectorizer(dtype=np.float32, sparse=False)
        categorical_vectorizer.fit(self.df[CATEGORICAL_COLUMNS].apply(dict, axis=1))
        return categorical_vectorizer
    
    def make_batch(self, data, max_len=None):
        """
        Creates a neural-network-friendly dict from the batch data.
        :param word_dropout: replaces token index with UNK_IDX with this probability
        :returns: a dict with {'title' : int64[batch, title_max_len]
        """
        batch = {}
        batch["Title"] = self.as_matrix(data["Title"].values, max_len=max_len)
        batch["FullDescription"] = self.as_matrix(data["FullDescription"].values, max_len=max_len)
        batch["Categorical"] = torch.tensor(self.categorical_vectorizer.transform(
            data[CATEGORICAL_COLUMNS].apply(dict, axis=1)
        ))

        if LOG_TARGET_COLUMN in data.columns:
            batch[LOG_TARGET_COLUMN] = torch.tensor(data[LOG_TARGET_COLUMN].values)

        return batch
    
    def get_iterators(self):
        data_train, data_val = train_test_split(self.df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        data_train.index = range(len(data_train))
        data_val.index = range(len(data_val))
        train_iterator = self.get_minibatches(data_train)
        valid_iterator = self.get_minibatches(data_val)
        return train_iterator, valid_iterator
           
    def get_minibatches(self, data, batch_size=BATCH_SIZE, shuffle=True, cycle=False, **kwargs):
        """get minibatches of data in random order"""
        logger.debug(f"The number of iterations: {len(data)//batch_size}")
        indices = np.arange(len(data))
        while True:
            if shuffle:
                indices = np.random.permutation(indices)
            for start in range(0, len(indices), batch_size):
                batch = self.make_batch(data.iloc[indices[start : start + batch_size]], **kwargs)
                target = batch.pop(LOG_TARGET_COLUMN)
                yield batch, target
        
            if not cycle:
                break
###############################################     model     ############################################################

# Initially, our FullDescription has a shape [batch_size, seq_len].
# After an Embedding layer shape will be [batch_size, seq_len, embedding_size].
# However, Conv1d layer expects batches of shape [batch_size, embedding_size, seq_len].
# We will use this layer to fix this misunderstanding.
class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))
    

class ThreeInputsNet(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_cat_features,
        hid_size=64,
    ):
        super().__init__()

        self.title_emb = nn.Embedding(n_tokens, hid_size)
        self.title_head = nn.Sequential(
            Reorder(),
            nn.Conv1d(hid_size, 2*hid_size, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(2*hid_size),
            nn.Conv1d(2*hid_size, 2*hid_size, kernel_size=(3)),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        self.desc_emb = nn.Embedding(n_tokens, hid_size)
        self.description_head = nn.Sequential(
            Reorder(),
            nn.Conv1d(hid_size, 2*hid_size, kernel_size=(3)),
            nn.ReLU(),
            nn.BatchNorm1d(2*hid_size),
            nn.Conv1d(2*hid_size, 2*hid_size, kernel_size=(3)),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.category_head = nn.Linear(n_cat_features, 2*hid_size)
        self.fc_out = nn.Linear(6*hid_size, 1)

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_emb = self.title_emb(input1)
        desc_emb = self.desc_emb(input2)

        title_out = self.title_head(title_emb)
        desc_out = self.description_head(desc_emb)
        category_out = self.category_head(input3)

        concatenated = torch.cat(
            [
                title_out.view(title_out.size(0), -1),
                desc_out.view(desc_out.size(0), -1),
                category_out.view(category_out.size(0), -1),
            ],
            dim=1,
        )
        out = self.fc_out(concatenated)
        return out.squeeze(1)


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
######################################################## train process  ###################################################

def train_step(
    model,
    train_iterator,
    criterion,
    optimizer,
    global_step,
    writer,
    device):
    model.train()
    train_loss = 0
    num_iter = 0
    for src_batch, trg in tqdm(train_iterator, desc='Train', leave=False):
        input = (src_batch['Title'].to(device), src_batch['FullDescription'].to(device), src_batch['Categorical'].to(device))
        trg = trg.to(device)
        output = model(input)
        loss = criterion(output, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        writer.add_scalar("Training/loss", loss.item(), global_step)
        global_step += 1
        num_iter += 1
        
    train_loss /= num_iter
    return round(train_loss, 3), global_step


def eval_step(
    model,
    valid_iterator,
    criterion,
    device
    ):
    model.eval()
    valid_loss = 0
    num_iter = 0
    with torch.no_grad():
        for src_batch, trg in tqdm(valid_iterator, desc="Validate", leave=False):
            input = (src_batch['Title'].to(device), src_batch['FullDescription'].to(device), src_batch['Categorical'].to(device))
            trg = trg.to(device)
            output = model(input)
            loss = criterion(output, trg)
            valid_loss += loss.item()
            num_iter += 1
    valid_loss /= num_iter
    return round(valid_loss, 3)


def train_model(
    model,
    dataset,
    criterion,
    optimizer,
    num_epochs,
    global_step,
    writer,
    device,
    ):
    last_global_step = 0
    for epoch in trange(num_epochs, desc="Epochs"):
        train_iterator, valid_iterator = dataset.get_iterators()
        train_loss, last_global_step = train_step(
            model,
            train_iterator,
            criterion,
            optimizer,
            last_global_step,
            writer,
            device
        )
        writer.add_scalar("Evaluation/train_loss", train_loss, epoch)
        logger.info(f"train loss: {round(train_loss, 3)}")

        valid_loss = eval_step(
            model,
            valid_iterator,
            criterion,
            device
        )
        writer.add_scalar("Evaluation/valid_loss", valid_loss, epoch)
        logger.info(f"valid loss: {valid_loss}")
        
    return train_loss, valid_loss
     
     
     
def main():
    start_time = datetime.now()
    logger.debug(f"Get dataiterators...")
    dataset = DataSet()
    #logger.debug(f"Number of batchs in train_dataloader: {len(train_iterator)}, in valid_dataloader: {len(valid_iterator)}")
    logger.debug(f"Build model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThreeInputsNet(n_tokens=len(dataset.vocab), n_cat_features=len(dataset.categorical_vectorizer.vocabulary_))
    model.to(device)
    logger.info(f"Using device is: {device}")
    num_params = count_model_parameters(model)
    logger.debug(f"Number of params for model: {num_params}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    logger.debug(f"Start train model...")
    train_loss, valid_loss = train_model(
        model,
        dataset,
        criterion,
        optimizer,
        num_epochs=NUM_EPOCHS,
        global_step=0,
        writer=SummaryWriter(log_dir=LOG_DIR),
        device=device,
        )
    end_time = datetime.now()
    logger.debug(f"Time execution: {end_time-start_time}, train_loss: {train_loss}, valid_loss: {valid_loss}")
    
if __name__ == "__main__":
    main()