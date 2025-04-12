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
import torch.nn.functional as F
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
NUM_EPOCHS = 10
CLIP = 1
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
        batch_size: int=128,
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


    def build_vocabs(self)-> Tuple[Vocab, Vocab]:
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

        special_tokens = [self.unk_token, self.sos_token, self.eos_token, self.pad_token]
        special_tokens_ids = [self.unk_token_id, self.sos_token_id, self.eos_token_id, self.pad_token_id]
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
        tokens = [self.sos_token] + self.tokenize_sentence(sentence) + [self.eos_token]
        return [vocab[token] for token in tokens]


    def collate_batch(self, batch: List[Tuple[str, str]])-> Tuple[torch.tensor, torch.tensor]:
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
        train_iter, valid_iter = Multi30k(root="../notebooks/.data/", split=("train", "valid"))
        train_data = list(train_iter)
        valid_data = list(valid_iter)
        logger.debug(f"The number of train_data: {len(train_data)}; number of valid_data: {len(valid_data)}")
        return train_data, valid_data
    
    def get_train_valid_dataloader(self)-> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True)
        valid_dataloader = DataLoader(self.valid_data, batch_size=self.batch_size, collate_fn=self.collate_batch)
        return train_dataloader, valid_dataloader
    
    
########################################### model ########################################################################

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = False, num_layers=n_layers)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]   
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(hidden))
        
        #outputs = [src len, batch size, enc hid dim]
        #hidden = [num_layers, batch size, dec hid dim]
        
        return outputs, hidden
    
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 1) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times for making possible concatenating along hidden_vectors
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        #energy = [batch size, src len, dec hid dim]
        
        attention = self.v(energy).squeeze(2)
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1) 
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 1) + emb_dim, dec_hid_dim, num_layers=n_layers)
        
        self.fc_out = nn.Linear((enc_hid_dim * 1) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [num_layers, batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden[-1,:, :], encoder_outputs)               
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src len, enc hid dim]
        
        weighted = torch.bmm(a, encoder_outputs) # batch matrix multiplication
        #weighted = [batch size, 1, enc hid dim]
        
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, batch size, (enc hid dim * 1) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden)
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len,  and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [n_layers, batch size, dec hid dim]
        #this also means that output == hidden
        # assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [batch size, output dim]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len-1, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t-1] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
    
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)

def build_model(num_src_tokens, num_trg_tokens, device):
    enc = Encoder(num_src_tokens, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, n_layers=2, dropout=0.5)
    attn = Attention(512, 512)
    dec = Decoder(num_trg_tokens, emb_dim=256, enc_hid_dim=512,  dec_hid_dim=512, n_layers=2, dropout=0.5, attention=attn)
    model = Seq2Seq(enc, dec, device)
    model.apply(init_weights)
    return model

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        output = model(src, trg)
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
            output = model(src, trg)
            output = output.view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            valid_loss += loss.item()
    valid_loss /= len(valid_dataloader)
    perplexity = round(torch.exp(torch.tensor(valid_loss)).item(), 3)
    return valid_loss, perplexity


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
        writer.add_scalar("Evaluation/train_perplexity", train_perplexity, epoch)
        logger.info(f"train loss: {round(train_loss, 3)}, perplexity: {train_perplexity}")

        valid_loss, valid_perplexity = eval_step(
            model,
            valid_dataloader,
            criterion,
            device
        )
        writer.add_scalar("Evaluation/valid_loss", valid_loss, epoch)
        writer.add_scalar("Evaluation/valid_perplexity", valid_perplexity, epoch)
        logger.info(f"valid loss: {train_perplexity}")
        
    return train_perplexity, valid_perplexity



def main():
    start_time = datetime.now()
    logger.debug(f"Get dataloaders...")
    dataset = DataSet()
    train_dataloader, valid_dataloader = dataset.get_train_valid_dataloader()
    logger.debug(f"Number of batchs in train_dataloader: {len(train_dataloader)}, in valid_dataloader: {len(valid_dataloader)}")
    logger.debug(f"Build model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(dataset.src_vocab), len(dataset.trg_vocab), device)
    model.to(device)
    logger.info(f"Using device is: {device}")
    num_params = count_model_parameters(model)
    logger.debug(f"Number of params for model: {num_params}")
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.trg_vocab["<pad>"])
    optimizer = torch.optim.Adam(model.parameters())
    
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
        writer=SummaryWriter(),
        device=device,
        )
    end_time = datetime.now()
    logger.debug(f"Time execution: {end_time-start_time}, train_perplexity: {train_perplexity}, valid_perplexity: {valid_perplexity}")
    
if __name__ == "__main__":
    main()