from datetime import datetime
from typing import List, Tuple, Dict
import logging
from collections import Counter
import random
from tqdm.auto import tqdm, trange
from nltk.tokenize import WordPunctTokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import vocab as Vocab
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(levelname)s: %(message)s')
logger = logging.getLogger("seq2seq")

######################################################  constants   ##########################################################
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

UNK_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3
NUM_EPOCHS = 15
CLIP = 1
MODEL_SAVE_PATH = "../models/seq2seq.pt"
TENSORBOARD_LOGS_DIR = "new_logs"
EVAL_TRANSLATE_SENT_IDX = [0, 100, 200, 300, 500, 600, 700, 800, 950]
MAX_EVAL_SENTENCE_LEN = 25
########################################################  data  ###############################################################


class DataSet:
    def __init__(
        self,
        min_freq: int = 2,
        unk_token: str = UNK_TOKEN,
        sos_token: str = SOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        pad_token: str = PAD_TOKEN,
        unk_token_id: int = UNK_TOKEN_ID,
        sos_token_id: int = SOS_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
        pad_token_id: int = PAD_TOKEN_ID,
        batch_size: int = 128,
    ):
        self.tokenizer = WordPunctTokenizer()
        self.min_freq = min_freq
        self.unk_token = UNK_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.unk_token_id = UNK_TOKEN_ID
        self.sos_token_id = SOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.batch_size = batch_size
        self.train_data, self.valid_data = self.get_train_valid_data()
        self.src_vocab, self.trg_vocab = self.build_vocabs()

    def tokenize_sentence(self, sentence: str) -> List[str]:
        return self.tokenizer.tokenize(sentence.lower().rstrip())

    def build_vocabs(self) -> Tuple[Vocab, Vocab]:
        """_summary_

        Args:
            train_data (List[Tuple]): list of tuples (src_sent, trg_sent)
            min_freq (int, optional): Minimum freq for word to add to vocab, defaults to 2.
            unk_token (str, optional): Unknown token, defaults to "<unk>".
            sos_token (str, optional): Start of sequence token, defaults to "<sos>".
            eos_token (str, optional): End of sequence token, defaults to "<eos>".
            pad_token (str, optional): padding token defaults to "<pad>".
        Return:
            src_vocab, trg_vocab
        """
        # build word_freqs for src and trg
        src_counter = Counter()
        trg_counter = Counter()
        for temp_src, temp_trg in self.train_data:

            src_counter.update(self.tokenize_sentence(temp_src))
            trg_counter.update(self.tokenize_sentence(temp_trg))

        # build vocab for src and trg
        src_vocab = Vocab(src_counter, min_freq=self.min_freq)
        trg_vocab = Vocab(trg_counter, min_freq=self.min_freq)

        special_tokens = [self.unk_token, self.sos_token,
                          self.eos_token, self.pad_token]
        special_tokens_ids = [
            self.unk_token_id, self.sos_token_id, self.eos_token_id, self.pad_token_id]
        for token, token_id in zip(special_tokens, special_tokens_ids):
            for vocab in (src_vocab, trg_vocab):
                if token not in vocab:
                    vocab.insert_token(token=token, index=token_id)
                vocab.set_default_index(0)
        return src_vocab, trg_vocab

    def encode_sentence(self, sentence: str, vocab: Vocab) -> list[int]:
        """_summary_

        Args:
            sentence (str): initial sentence
            vocab (torchtext.vocab): vocab of tokens

        Returns:
            list[int]: list of tokens ids
        """
        tokens = [self.sos_token] + \
            self.tokenize_sentence(sentence) + [self.eos_token]
        return [vocab[token] for token in tokens]

    def collate_batch(self, batch: List[Tuple[str, str]]) -> Tuple[torch.tensor, torch.tensor]:
        src_lst, trg_lst = [], []
        for src_sent, trg_sent in batch:
            encode_src = self.encode_sentence(src_sent, self.src_vocab)[::-1]
            encode_trg = self.encode_sentence(trg_sent, self.trg_vocab)
            src_lst.append(torch.tensor(encode_src))
            trg_lst.append(torch.tensor(encode_trg))
        src_padded = pad_sequence(src_lst, padding_value=self.pad_token_id)
        trg_padded = pad_sequence(trg_lst, padding_value=self.pad_token_id)
        return src_padded, trg_padded

    @staticmethod
    def get_train_valid_data():
        logger.debug(f"Start loading train and valid data...")
        train_iter, valid_iter = Multi30k(
            root="../notebooks/.data/", split=("train", "valid"))
        train_data = list(train_iter)
        valid_data = list(valid_iter)
        logger.debug(
            f"The number of train_data: {len(train_data)}; number of valid_data: {len(valid_data)}")
        return train_data, valid_data

    def get_train_valid_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True)
        valid_dataloader = DataLoader(
            self.valid_data, batch_size=self.batch_size, collate_fn=self.collate_batch)
        return train_dataloader, valid_dataloader


########################################################  model  ###############################################################

class Encoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.n_tokens = n_tokens
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_tokens, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, src):
        # src has a shape of [seq_len, batch_size]
        embedded = self.embedding(src)  # batch_size, seq_len, emb_dim
        embedded = self.dropout(embedded)
        _, hidden = self.rnn(embedded)

        return hidden


class Decoder(nn.Module):
    def __init__(self, n_tokens, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.n_tokens = n_tokens
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_tokens, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hid_dim, n_tokens)

    def forward(self, input, hidden):
        # input has a shape of [batch_size]
        # hidden is a tuple of two tensors:
        # 1) hidden state
        # 2) cell state
        # both of shape [n_layers, batch_size, hid_dim]
        # (n_directions in the decoder shall always be 1)
        input = input.unsqueeze(dim=0)
        embedded = self.embedding(input)
        hidden = (hidden[0].contiguous(), hidden[1].contiguous())
        output, hidden = self.rnn(embedded, hidden)
        pred = self.fc(output.squeeze(0))
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.hid_dim, "encoder and decoder must have same hidden dim"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "encoder and decoder must have equal number of layers"

    def forward(self, src_batch, trg_batch, device, teacher_forcing_ratio=0.5):
        # src_batch: (batch_size, src_length)
        # trg_batch: (batch_size, trg_length)
        trg_length, batch_size = trg_batch.shape
        preds = []
        hidden = self.encoder(src_batch)
        input = trg_batch[0, :]
        for i in range(1, trg_length):
            temp_pred, hidden = self.decoder(input, hidden)
            preds.append(temp_pred)
            teacher_force = random.random() < teacher_forcing_ratio
            _, top_pred = temp_pred.max(dim=1)
            input = trg_batch[i, :] if teacher_force else top_pred
        return torch.stack(preds)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


def build_model(num_src_tokens, num_trg_tokens):
    enc = Encoder(num_src_tokens, emb_dim=256,
                  hid_dim=512, n_layers=2, dropout=0.5)
    dec = Decoder(num_trg_tokens, emb_dim=256,
                  hid_dim=512, n_layers=2, dropout=0.5)
    model = Seq2Seq(enc, dec)
    model.apply(init_weights)
    return model


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def translate_sentence(src_sentence: str, dataset=Dataset, model=Seq2Seq, device="cpu") -> str:
    with torch.no_grad():
        input_tensor = torch.tensor(dataset.encode_sentence(
            src_sentence, dataset.src_vocab)[::-1]).to(device).unsqueeze(1)
        hidden = model.encoder(input_tensor)
        inputs = [dataset.trg_vocab.get_stoi()[SOS_TOKEN]]
        for i in range(MAX_EVAL_SENTENCE_LEN):
            input_tensor = torch.LongTensor([inputs[-1]]).to(device)
            temp_pred, hidden = model.decoder(input_tensor, hidden)
            predicted_token = temp_pred.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == dataset.trg_vocab.get_stoi()[EOS_TOKEN]:
                break
        tokens = dataset.trg_vocab.lookup_tokens(inputs)
        if tokens[-1] == EOS_TOKEN:
            predict_sentense = ' '.join(tokens[1:-1])
        else:
            predict_sentense = ' '.join(tokens[1:])
    return predict_sentense


######################################################## train process  ###################################################


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
    for src, trg in tqdm(train_dataloader, desc='Train', leave=False):
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, device=device)
        output = output.view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        writer.add_scalar("Training/loss", loss.item(), global_step)
        global_step += 1

    train_loss /= len(train_dataloader)
    perplexity = round(torch.exp(torch.tensor(train_loss)).item(), 3)
    return train_loss, perplexity, global_step


def eval_step(
    model,
    valid_dataloader,
    criterion,
    device
):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for src, trg in tqdm(valid_dataloader, desc="Validate", leave=False):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, device=device)
            output = output.view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            valid_loss += loss.item()
    valid_loss /= len(valid_dataloader)
    perplexity = round(torch.exp(torch.tensor(valid_loss)).item(), 3)
    return valid_loss, perplexity


def translate_test_sentences(eval_translate_sents_idx: List[int], dataset: Dataset, model: Seq2Seq, writer: SummaryWriter, device="cpu"):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    logger.info("Some examples of translation senetences:")
    text = ""
    for i in eval_translate_sents_idx:
        predict_log = f"predict sentence: {translate_sentence(dataset.valid_data[i][0], dataset, model, device)}\n"
        test_log = f"test sentence: {dataset.valid_data[i][1]}\n\n\n"
        logger.info(predict_log)
        logger.info(test_log)
        text += predict_log
        text += test_log
    
    writer.add_text("Example of translation", text_string=text)


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
    best_perplexity = float('inf')
    for epoch in trange(num_epochs, desc="Epochs"):
        train_loss, train_perplexity, last_global_step = train_step(
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
        writer.add_scalar("Evaluation/train_perplexity",
                          train_perplexity, epoch)
        logger.info(
            f"train loss: {round(train_loss, 3)}, perplexity: {train_perplexity}")

        valid_loss, valid_perplexity = eval_step(
            model,
            valid_dataloader,
            criterion,
            device
        )
        if valid_perplexity < best_perplexity:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_perplexity = valid_perplexity

        writer.add_scalar("Evaluation/valid_loss", valid_loss, epoch)
        writer.add_scalar("Evaluation/valid_perplexity",
                          valid_perplexity, epoch)
        logger.info(f"valid loss: {train_perplexity}")
    logger.info(
        f"The best model with perplexity: {best_perplexity} saved at {MODEL_SAVE_PATH}!!!")
    return train_perplexity, valid_perplexity


def main():
    start_time = datetime.now()
    logger.debug(f"Get dataloaders...")
    dataset = DataSet()
    train_dataloader, valid_dataloader = dataset.get_train_valid_dataloader()
    logger.debug(
        f"Number of batchs in train_dataloader: {len(train_dataloader)}, in valid_dataloader: {len(valid_dataloader)}")
    logger.debug(f"Build model...")
    model = build_model(len(dataset.src_vocab), len(dataset.trg_vocab))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Using device is: {device}")
    num_params = count_model_parameters(model)
    logger.debug(f"Number of params for model: {num_params}")
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.trg_vocab["<pad>"])
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()
    logger.debug(f"Start train model...")
    train_perplexity, valid_perplexity = train_model(
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
    translate_test_sentences(EVAL_TRANSLATE_SENT_IDX, dataset=dataset, model=model, writer=writer, device=device)
    end_time = datetime.now()
    logger.debug(
        f"Time execution: {end_time-start_time}, train_perplexity: {train_perplexity}, valid_perplexity: {valid_perplexity}")


if __name__ == "__main__":
    main()

