#####################################################        import           #########################################

from typing import List, Optional
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


#####################################################       constants         ##########################################

FILE_PATH = "../data/reviews.csv"
TEXT_COLUMNS = ["Title", "FullDescription"]
CATEGORICAL_COLUMNS  = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
TARGET_COLUMN = "SalaryNormalized"
LOG_TARGET_COLUMN = "Log1pSalary"
MIN_TOKEN_COUNT = 10
TEST_SIZE = 0.2 
RANDOM_STATE = 42
UNK = "UNK"
PAD = "PAD"



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
        return matrix
    
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
        batch["Categorical"] = self.categorical_vectorizer.transform(
            data[CATEGORICAL_COLUMNS].apply(dict, axis=1)
        )

        if LOG_TARGET_COLUMN in data.columns:
            batch[LOG_TARGET_COLUMN] = data[LOG_TARGET_COLUMN].values

        return batch
    
    def get_iterators(self):
        data_train, data_val = train_test_split(self.df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        data_train.index = range(len(data_train))
        data_val.index = range(len(data_val))
        train_iterator = self.iterate_minibatches(data_train)
        valid_iterator = self.iterate_minibatches(data_val)
        return train_iterator, valid_iterator
           
    def iterate_minibatches(self, data, batch_size=256, shuffle=True, cycle=False, **kwargs):
        """iterates minibatches of data in random order"""
        while True:
            indices = np.arange(len(data))
            if shuffle:
                indices = np.random.permutation(indices)

            for start in range(0, len(indices), batch_size):
                batch = self.make_batch(data.iloc[indices[start : start + batch_size]], **kwargs)
                target = batch.pop(LOG_TARGET_COLUMN)
                yield batch, target

            if not cycle:
                break
        
        
