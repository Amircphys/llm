# Import required libraries
import os               # For file and path operations (check_file_exists, extract_dataset)
import urllib.request   # For downloading dataset files from URLs
import tarfile          # For extracting .tar.gz dataset archives
import torch            # Main PyTorch library for tensor operations and deep learning
import torch.nn as nn   # Neural network modules, layers, and utilities
from torch.utils.data import DataLoader, IterableDataset  # For efficient data loading and streaming
import random           # For setting random seeds in reproducibility
from tqdm import tqdm   # For progress bars in training and evaluation
import math             # For computing perplexity using exp()
import re               # For preprocessing text (replacing numbers with placeholders)
from transformers import AutoTokenizer # For loading a pre-trained tokenizer
from loggers import strim_logger

SEED = 42
TOKENIZER_MODEL = "microsoft/Phi-3.5-mini-instruct"
DATA_URL = "https://www.thelmbook.com/data/news"
EVAL_INTERVAL = 200_000
EXAMPLES_PROCESSED = 0  # Counter for tracking progress toward next evaluation
MAX_LENGTH = 50
MODEL_NAME = "RNN_LM"
EMB_DIM = 128        # Embedding dimension
NUM_LAYERS = 2        # Number of RNN layers
BATCH_SIZE = 64     # Training batch size
LEARNING_RATE = 0.001 # Learning rate for optimization
NUM_EPOCHS = 5      # Number of training epochs
CONTEXT_SIZE = 30  # Maximum input sequence length

CONTEXT = [
        "Moscow",
        "New York",
        "The hurricane",
        "The President"
    ]

device = "cuda" if torch.cuda.is_available() else "cpu"
# 1.  Utility functions and classes

def set_seed(seed):
    """
    Sets random seeds for reproducibility across different Python libraries.
    This ensures that random operations give the same results across runs.

    Args:
        seed (int): Seed value for random number generation
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for PyTorch's CPU random number generator
    torch.manual_seed(seed)
    # Set seed for PyTorch's GPU random number generator
    torch.cuda.manual_seed_all(seed)
    # Requests cuDNN to use deterministic algorithms when possible
    # Note: This may impact performance and might not guarantee determinism in all cases
    torch.backends.cudnn.deterministic = True
    # Disables cuDNN's auto-tuner which finds the best algorithm for your specific input size
    # Ensures consistent behavior but might be slower as it doesn't optimize for input sizes
    torch.backends.cudnn.benchmark = False

def check_file_exists(filename):
    """
    Checks if a file exists in the current directory.
    Args:
        filename (str): Name of the file to check
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.exists(filename)

def create_collate_fn(tokenizer):
    """
    Creates a collate function for batching sequences of different lengths.
    This function pads shorter sequences to match the longest sequence in the batch.

    Args:
        tokenizer: Tokenizer object containing padding token information

    Returns:
        function: Collate function that handles padding in batches
    """
    def collate_fn(batch):
        # Separate inputs and targets from batch
        input_seqs, target_seqs = zip(*batch)
        # Get padding token ID from tokenizer
        pad_index = tokenizer.pad_token_id
        # Pad input sequences to same length
        input_padded = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=pad_index)
        # Pad target sequences to same length
        target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=pad_index)
        return input_padded, target_padded
    return collate_fn

def download_file(url):
    """
    Downloads a file from the given URL if it doesn't exist locally.
    Uses a custom User-Agent to help prevent download blocks.

    Args:
        url (str): URL of the file to download
    Returns:
        str: Name of the downloaded file ("news.tar.gz")
    """
    # Always use news.tar.gz as the filename, regardless of URL
    filename = "news.tar.gz"

    if not check_file_exists(filename):
        strim_logger.info(f"Downloading dataset from {url}...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            with open(filename, "wb") as out_file:
                out_file.write(response.read())
        strim_logger.info("Download completed.")
    else:
        strim_logger.info(f"{filename} already downloaded.")
    return filename

def is_within_directory(directory, target):
    """
    Checks if a target path is within a specified directory by comparing absolute paths.

    Args:
        directory (str): Base directory path
        target (str): Target path to check
    Returns:
        bool: True if target's absolute path starts with directory's absolute path
    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory

def extract_dataset(filename):
    """
    Extracts train.txt and test.txt from the downloaded archive.
    Includes debug information about archive contents.

    Args:
        filename (str): Name of the archive file
    Returns:
        tuple: Paths to extracted train and test files
    """
    data_dir = os.path.join(os.path.dirname(filename), "news")
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")

    if check_file_exists(train_path) and check_file_exists(test_path):
        strim_logger.info("Data files already extracted.")
        return train_path, test_path

    strim_logger.info("\nListing archive contents:")
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar.getmembers():
            strim_logger.info(f"Archive member: {member.name}")

        strim_logger.info("\nExtracting files...")
        # Extract to current directory first
        tar.extractall('.')

    if not (check_file_exists(train_path) and check_file_exists(test_path)):
        raise FileNotFoundError(f"Required files not found in the archive. Please check the paths above.")

    strim_logger.info("Extraction completed.")
    return train_path, test_path


class IterableTextDataset(IterableDataset):
    """
    An iterable dataset for processing text data in a memory-efficient way.
    Instead of loading all data into memory, it streams data from disk.
    Inherits from PyTorch's IterableDataset for streaming support.

    Args:
        file_path (str): Path to the text file containing sentences
        tokenizer: Tokenizer object for converting text to tokens
        max_length (int): Maximum sequence length to process (default: 30)
    """
    def __init__(self, file_path, tokenizer, max_length=30):
        # Store file path for reading data
        self.file_path = file_path
        # Store tokenizer for text processing
        self.tokenizer = tokenizer
        # Set maximum sequence length to truncate long sequences
        self.max_length = max_length
        self._count_sentences()
    
    def _count_sentences(self):
        strim_logger.info(f"Counting sentences in {self.file_path}...")
        with open(self.file_path, 'r', encoding="utf-8") as f:
            self._num_sentences = sum(1 for _ in f)
        strim_logger.info(f"Found {self._num_sentences} sentences in {self.file_path}.")
        
    def __len__(self):
        return self._num_sentences
    
    def __iter__(self):
        """
        Creates an iterator over the dataset.
        This method is called when iterating over the dataset.

        Yields:
            tuple: (input_sequence, target_sequence) pairs for language modeling
                  input_sequence is the sequence up to the last token
                  target_sequence is the sequence shifted one position right
        """
        # Open file in read mode with UTF-8 encoding
        with open(self.file_path, 'r', encoding="utf-8") as f:
            # Process each line (sentence) in the file
            for line in f:
                # Remove leading/trailing whitespace
                sentence = line.strip()
                # Replace all numbers with ### placeholder
                # This reduces vocabulary size and helps model generalize
                sentence = re.sub(r"\d+", "###", sentence)

                # Convert sentence to token IDs
                encoded_sentence = self.tokenizer.encode(
                    sentence,
                    max_length=self.max_length,
                    truncation=True
                )

                # Only use sequences with at least 2 tokens
                # (need at least one input and one target token)
                if len(encoded_sentence) >= 2:
                    # Input is all tokens except last
                    input_seq = encoded_sentence[:-1]
                    # Target is all tokens except first
                    target_seq = encoded_sentence[1:]
                    # Convert to PyTorch tensors and yield
                    yield torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
                    
def create_datasets(train_file, test_file, tokenizer, max_length=30):
    """
    Creates IterableTextDataset objects for training and testing.
    These datasets will stream data from disk instead of loading it all into memory.

    Args:
        train_file (str): Path to training data file
        test_file (str): Path to test data file
        tokenizer: Tokenizer object for text processing

    Returns:
        tuple: (train_dataset, test_dataset) - Dataset objects for training and testing
    """
    # Create training dataset
    train_dataset = IterableTextDataset(train_file, tokenizer, max_length)
    # Create test dataset
    test_dataset = IterableTextDataset(test_file, tokenizer, max_length)

    # Print dataset sizes
    strim_logger.info(f"Training sentences: {len(train_dataset)}")
    strim_logger.info(f"Test sentences: {len(test_dataset)}")

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn):
    """
    Creates DataLoader objects for efficient data iteration.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size (int): Number of sequences per batch
        collate_fn: Function to handle padding and batch creation

    Returns:
        tuple: (train_dataloader, test_dataloader) - DataLoader objects for
               iterating over batches of data with proper padding
    """
    # Create training data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,    # Function to handle padding
        num_workers=0             # Number of worker processes (0 = single process)
    )
    # Create test data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    return train_dataloader, test_dataloader

def download_and_prepare_data(url, batch_size, tokenizer, max_length=30):
    """
    Main function to handle the complete data preparation pipeline.
    Downloads data, extracts it, and creates necessary dataset objects.

    Args:
        url (str): URL where the dataset archive can be downloaded
        batch_size (int): Batch size for data loading
        tokenizer: Tokenizer object for text processing
        max_length (int): Maximum sequence length for tokenization (default: 30)

    Returns:
        tuple: (train_dataloader, test_dataloader) - Ready-to-use data loaders
    """
    # Step 1: Download dataset archive from URL
    #filename = download_file(url)
    filename = "news"
    # Step 2: Extract training and test files from archive
    train_file, test_file = extract_dataset(filename)

    # Step 3: Create dataset objects for streaming data
    train_dataset, test_dataset = create_datasets(train_file, test_file, tokenizer, max_length)

    # Step 4: Create function to handle batch creation
    collate_fn = create_collate_fn(tokenizer)

    # Step 5: Create and return data loaders
    return create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn)


def compute_loss_and_perplexity(model, dataloader, tokenizer, criterion, device, max_sentences=1000):
    """
    Evaluates model performance by computing loss and perplexity on data.

    Args:
        model (nn.Module): The language model to evaluate
        dataloader (DataLoader): Data loader containing batched sequences
        tokenizer: Tokenizer for handling special tokens like padding
        criterion: Loss function (usually CrossEntropyLoss)
        device: Device to run computation on (cuda/cpu)
        max_sentences (int): Maximum number of sentences to evaluate (default: 1000)
                           Limits evaluation to a subset for faster validation

    Returns:
        tuple: (average_loss, perplexity)
               - average_loss: Mean loss per token (excluding padding)
               - perplexity: exp(average_loss), lower is better
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Initialize counters for loss calculation
    total_loss = 0.0          # Accumulator for total loss across all batches
    total_tokens = 0          # Counter for total number of tokens (excluding padding)
    sentences_processed = 0    # Counter for number of sentences processed

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate through data with progress bar
        for input_seq, target_seq in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move input and target sequences to specified device
            input_seq = input_seq.to(device)      # Shape: (batch_size, seq_len)
            target_seq = target_seq.to(device)    # Shape: (batch_size, seq_len)

            # Get current batch size (might be smaller for last batch)
            batch_size_current = input_seq.size(0)

            # Forward pass through the model
            logits = model(input_seq)             # Shape: (batch_size, seq_len, vocab_size)

            # Reshape logits and target for loss calculation
            logits = logits.reshape(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            target = target_seq.reshape(-1)              # Shape: (batch_size * seq_len)

            # Create mask to exclude padding tokens
            mask = target != tokenizer.pad_token_id

            # Compute loss only on non-padded tokens
            loss = criterion(logits[mask], target[mask])

            # Update counters
            loss_value = loss.item() * mask.sum().item()  # Total loss for this batch
            total_loss += loss_value                      # Accumulate batch loss
            total_tokens += mask.sum().item()             # Count non-padding tokens

            # Update sentence counter and check if we've reached maximum
            sentences_processed += batch_size_current
            if sentences_processed >= max_sentences:
                break

    # Calculate final metrics
    average_loss = total_loss / total_tokens           # Normalize loss by number of tokens
    perplexity = math.exp(average_loss)               # Convert loss to perplexity

    return average_loss, perplexity

def perform_model_evaluation(model, test_dataloader, criterion, tokenizer, device, contexts):
    """
    Perform evaluation of the model including loss calculation, perplexity, and text generation.

    Args:
        model: The neural network model
        test_dataloader: DataLoader containing test/validation data
        criterion: Loss function
        tokenizer: Tokenizer for text generation
        device: Device to run computations on (cuda/cpu)
        contexts: List of context strings for text generation

    Returns:
        tuple: (average_loss, perplexity)
    """
    # Switch to evaluation mode
    model.eval()

    # Compute metrics
    average_loss, perplexity = compute_loss_and_perplexity(
        model, test_dataloader, tokenizer, criterion, device, max_sentences=1000
    )

    strim_logger.info(f"Validation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}")

    # Generate text using the contexts
    strim_logger.info("Generating text based on contexts using generate_text:\n")
    for context in contexts:
        generated_text = generate_text(
            model=model,          # The loaded language model
            start_string=context, # Context to continue
            tokenizer=tokenizer,  # Tokenizer for text conversion
            device=device,        # CPU or GPU device
            max_length=50         # Maximum length of generated sequence
        )
        strim_logger.info(f"\nContext: {context}")
        strim_logger.info(f"\nGenerated text: {generated_text}\n")

    return average_loss, perplexity


def generate_text(model, start_string, tokenizer, device, max_length=50):
    """
    Generates text continuation from a given start string using greedy decoding.
    This method always chooses the most likely next token.

    Args:
        model (nn.Module): Trained language model
        start_string (str): Initial text to continue from
        tokenizer: Tokenizer for text processing
        device: Device to run generation on (cuda/cpu)
        max_length (int): Maximum length of generated sequence

    Returns:
        str: Generated text continuation
    """
    # Set model to evaluation mode
    model.eval()

    # Convert start string to token ids and move to device
    # return_tensors="pt" returns PyTorch tensor instead of list
    tokens = tokenizer.encode(start_string, return_tensors="pt", max_length=max_length, truncation=True).to(device)

    # Initialize generated sequence with input tokens
    generated = tokens

    # Generate new tokens one at a time
    for _ in range(max_length):
        # Get model's predictions
        output = model(generated)                    # Shape: (1, seq_len, vocab_size)
        # Get logits for the next token (last position)
        next_token_logits = output[0, -1, :]        # Shape: (vocab_size)

        # Choose token with highest probability (greedy decoding)
        # unsqueeze twice to match expected shape (1, 1)
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)

        # Add new token to generated sequence
        generated = torch.cat((generated, next_token_id), dim=1)

        # Stop if end of sequence token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Convert token ids back to text
    generated_text = tokenizer.decode(generated.squeeze().tolist())
    return generated_text


def save_model(model, tokenizer, file_prefix):
    model_state = {
        "state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "emb_dim": model.emb_dim,
        "num_layers": model.num_layers,
        "pad_index": model.pad_index,
        "training": model.training  # Save training state
    }

    torch.save(model_state, f"{file_prefix}_model.pth")
    tokenizer.save_pretrained(f"{file_prefix}_tokenizer")
    
def load_model(file_prefix):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load state dict to the correct device first
    model_state = torch.load(f"{file_prefix}_model.pth", map_location=device, weights_only=True)

    # Create model and move it to device before loading state dict
    model = RecurrentLanguageModel(
        model_state["vocab_size"],
        model_state["emb_dim"],
        model_state["num_layers"],
        model_state["pad_index"]
    ).to(device)

    # Load state dict after model is on correct device
    model.load_state_dict(model_state["state_dict"])

    # Keep model on device
    model.eval()  # Set to evaluation mode

    tokenizer = AutoTokenizer.from_pretrained(f"{file_prefix}_tokenizer")
    return model, tokenizer


# ----------------------------
# Recurrent Language Model Class
# ----------------------------

def initialize_weights(model):
    """
    Initializes model weights using Xavier uniform initialization for multi-dimensional
    parameters and uniform initialization for biases and other 1D parameters.

    Args:
        model (nn.Module): PyTorch model whose weights need to be initialized
    """
    # Loop through all named parameters in the model
    for name, param in model.named_parameters():
        # Check if parameter has more than 1 dimension (e.g., weight matrices)
        if param.dim() > 1:
            # Use Xavier uniform initialization for weight matrices
            # This helps prevent vanishing/exploding gradients by keeping the variance constant
            nn.init.xavier_uniform_(param)
        else:
            # For 1D parameters (like biases), use simple uniform initialization
            nn.init.uniform_(param)

class ElmanRNNUnit(nn.Module):
    """
    Implementation of a single Elman RNN unit (a simple recurrent neural network cell).
    This is the basic building block of our RNN that processes one time step of input.

    Args:
        emb_dim (int): Dimension of the embedding/hidden state vectors
    """
    def __init__(self, emb_dim):
        super(ElmanRNNUnit, self).__init__()
        # Hidden-to-hidden weight matrix: transforms previous hidden state
        # Shape: (emb_dim, emb_dim)
        self.Uh = nn.Parameter(torch.rand(emb_dim, emb_dim))

        # Input-to-hidden weight matrix: transforms current input
        # Shape: (emb_dim, emb_dim)
        self.Wh = nn.Parameter(torch.rand(emb_dim, emb_dim))

        # Bias term added to the sum of transformations
        # Shape: (emb_dim,)
        self.b = nn.Parameter(torch.rand(emb_dim))

    def forward(self, x, h):
        """
        Computes one step of the RNN unit.

        Args:
            x (torch.Tensor): Current input tensor of shape (batch_size, emb_dim)
            h (torch.Tensor): Previous hidden state of shape (batch_size, emb_dim)

        Returns:
            torch.Tensor: New hidden state of shape (batch_size, emb_dim)

        The formula implemented is: h_new = tanh(x @ Wh + h @ Uh + b)
        where @ represents matrix multiplication
        """
        # 1. Transform current input: x @ Wh
        input_transform = x @ self.Wh

        # 2. Transform previous hidden state: h @ Uh
        hidden_transform = h @ self.Uh

        # 3. Add both transformations and bias
        # 4. Apply tanh activation function to get new hidden state
        # tanh squashes values to range (-1, 1), helping prevent exploding gradients
        return torch.tanh(input_transform + hidden_transform + self.b)

class ElmanRNN(nn.Module):
    """
    Multi-layer Elman RNN implementation that processes entire sequences.
    Stacks multiple RNN units to create a deeper network that can learn more complex patterns.

    Args:
        emb_dim (int): Dimension of embeddings and hidden states
        num_layers (int): Number of stacked RNN layers
    """
    def __init__(self, emb_dim, num_layers):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        # Create a list of RNN units, one for each layer
        # ModuleList is used so PyTorch tracks all parameters
        self.rnn_units = nn.ModuleList(
            [ElmanRNNUnit(emb_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        """
        Processes input sequence through all RNN layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim)
        """
        # Get dimensions from input tensor
        batch_size, seq_len, emb_dim = x.size()

        # Initialize hidden states for each layer with zeros
        # Each hidden state has shape (batch_size, emb_dim)
        h_prev = [
            torch.zeros(batch_size, emb_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        # Will store outputs for each time step
        outputs = []

        # Process each time step
        for t in range(seq_len):
            # Get input for current time step
            input_t = x[:, t]

            # Process through each layer
            for l, rnn_unit in enumerate(self.rnn_units):
                # Compute new hidden state for this layer
                h_new = rnn_unit(input_t, h_prev[l])

                # Update hidden state for this layer
                h_prev[l] = h_new

                # Output of this layer becomes input to next layer
                input_t = h_new

            # Add final layer's output to results
            outputs.append(input_t)

        # Stack all time steps' outputs into a single tensor
        # Shape: (batch_size, seq_len, emb_dim)
        return torch.stack(outputs, dim=1)

class RecurrentLanguageModel(nn.Module):
    """
    Complete language model implementation combining embedding layer,
    multi-layer RNN, and output projection layer.

    The model architecture is:
    1. Input tokens -> Embedding Layer -> Embedded Vectors
    2. Embedded Vectors -> RNN Layers -> Context Vectors
    3. Context Vectors -> Linear Layer -> Vocabulary Predictions

    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens)
        emb_dim (int): Dimension of embeddings and hidden states
        num_layers (int): Number of RNN layers
        pad_index (int): Index used for padding tokens
    """
    def __init__(self, vocab_size, emb_dim, num_layers, pad_index):
        super().__init__()

        # Save model parameters
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.pad_index = pad_index

        # Embedding layer: converts token indices to dense vectors
        # pad_index tokens will be mapped to zero vectors
        self.embedding = nn.Embedding(vocab_size, emb_dim, pad_index)

        # RNN layers for processing sequences
        self.rnn = ElmanRNN(
            emb_dim=emb_dim,
            num_layers=num_layers
        )

        # Final linear layer to convert RNN outputs to vocabulary predictions
        # Output size is vocab_size to get logits for each possible token
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """
        Processes input sequences through the entire model.

        Args:
            x (torch.Tensor): Input tensor of token indices
                            Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Output logits for next token prediction
                         Shape: (batch_size, seq_len, vocab_size)

        Process:
        1. Convert token indices to embeddings
        2. Process embeddings through RNN layers
        3. Project RNN outputs to vocabulary size
        """
        # Convert token indices to embeddings
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        embeddings = self.embedding(x)

        # Process through RNN layers
        # Shape: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        rnn_output = self.rnn(embeddings)

        # Project to vocabulary size to get logits
        # Shape: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.fc(rnn_output)

        return logits
    
def get_model(vocab_size, emb_dim, num_layers, tokenizer):
    # Initialize the RNN language model with specified architecture parameters
    # vocab_size: determines output layer dimensionality
    # emb_dim: size of word embeddings and hidden states
    # num_layers: number of RNN layers
    # pad_token_id: special token ID used for padding shorter sequences
    model = RecurrentLanguageModel(vocab_size, emb_dim, num_layers, tokenizer.pad_token_id)

    # Move the model to GPU if available
    model.to(device)

    # Initialize model weights using custom initialization scheme
    # This is important for stable training of deep neural networks
    initialize_weights(model)

    # Calculate and display the total number of trainable parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    strim_logger.info(f"Total trainable parameters: {total_params}\n")
    return model
    
def train_model(
    model, 
    optimizer,
    criterion,
    num_epochs, 
    train_dataloader,
    test_dataloader, 
    context, 
    tokenizer,
    eval_interval,
    max_length,
    model_name,
    ):
    
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        strim_logger.info(f"Starting Epoch {epoch+1}/{num_epochs}, Model in training mode: {model.training}")

        # Initialize tracking variables for this epoch
        total_loss = 0.0      # Accumulator for loss across all batches
        total_tokens = 0      # Counter for actual tokens processed (excluding padding)
        # Create progress bar for monitoring training progress
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Iterate through batches in the training data
        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            # Move input and target sequences to GPU if available
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            # Get current batch dimensions for reshaping operations
            batch_size_current, seq_len = input_seq.shape

            # Clear gradients from previous batch
            # This is necessary as PyTorch accumulates gradients by default
            optimizer.zero_grad()

            # Forward pass: get model predictions for this batch
            # output shape: (batch_size, seq_len, vocab_size)
            output = model(input_seq)

            # Reshape output and target tensors for loss computation
            # - output: reshape to (batch_size * seq_len, vocab_size) for CrossEntropyLoss
            # - target: reshape to (batch_size * seq_len) to match CrossEntropyLoss requirements
            output = output.reshape(batch_size_current * seq_len, vocab_size)
            target = target_seq.reshape(batch_size_current * seq_len)

            # Count number of non-padding tokens in target
            # This is needed because to calculate the average loss for multiple batches we need to divide the total loss
            # by the number of tokens in these batches, but criterion(output, target) returns the average loss per token in a batch.
            # So, we will multiply the loss per token by the number of tokens to get the loss per batch
            non_padding_token_count = (target != tokenizer.pad_token_id).sum().item()

            # Compute loss between model predictions and actual targets
            loss = criterion(output, target)
            # Backward pass: compute gradients of loss with respect to model parameters
            loss.backward()
            # Update model parameters using calculated gradients
            optimizer.step()

            # Calculate actual loss value for this batch
            # Multiply the loss per token by number of non-padding tokens to get total loss for the batch
            loss_value = loss.item() * non_padding_token_count
            # Accumulate total loss for epoch statistics
            total_loss += loss_value
            # Accumulate total number of actual tokens processed
            total_tokens += non_padding_token_count
            # Increment counter for examples processed
            examples_processed += batch_size_current

            # Update progress bar with current batch loss
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Periodic evaluation after processing specified number of examples
            if examples_processed >= eval_interval:
                # Calculate average loss over the last eval_interval examples
                avg_loss = total_loss / total_tokens
                strim_logger.info(f"\nAfter {examples_processed} examples, Average Loss: {avg_loss:.4f}")

                # Switch to evaluation mode
                model.eval()

                # Compute validation metrics
                # average_loss: mean loss on validation set
                # perplexity: exponential of average loss, lower is better
                # sentences_processed: number of validation sequences evaluated
                average_loss, perplexity = compute_loss_and_perplexity(
                    model, test_dataloader, tokenizer, criterion, device, max_sentences=1000
                )
                strim_logger.info(f"Validation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}")

                # Generate sample texts to qualitatively assess model performance
                strim_logger.info("Generating text based on contexts using generate_text:\n")
                for context in contexts:
                    # Generate text continuation for each test context
                    generated_text = generate_text(
                        model=model,          # The loaded language model
                        start_string=context, # Context to continue
                        tokenizer=tokenizer,  # Tokenizer for text conversion
                        device=device,        # CPU or GPU device
                        max_length=max_length         # Maximum length of generated sequence
                    )
                    strim_logger.info(f"\nContext: {context}")
                    strim_logger.info(f"\nGenerated text: {generated_text}")

                # Switch back to training mode for continued training
                model.train()
                # Reset counters for next evaluation interval
                examples_processed = 0
                total_loss = 0.0
                total_tokens = 0

        # End-of-epoch reporting
        if total_tokens > 0:
            # Calculate and display average loss for the epoch
            avg_loss = total_loss / total_tokens
            strim_logger.info(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        else:
            # Handle edge case where no tokens were processed
            strim_logger.info(f"\nEpoch {epoch+1}/{num_epochs} completed.")

        # Perform end-of-epoch validation
        model.eval()
        average_loss, perplexity = compute_loss_and_perplexity(
            model, test_dataloader, tokenizer, criterion, device, max_sentences=1000
        )
        strim_logger.info(f"Validation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}\n")

        strim_logger.info("Generating text based on contexts using generate_text:\n")
        for context in contexts:
            # Generate text continuation for each test context
            generated_text = generate_text(
                model=model,
                start_string=context,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length  # Limit generation to 50 tokens for brevity
            )
            strim_logger.info(f"\nContext: {context}")
            strim_logger.info(f"\nGenerated text: {generated_text}")

        # Reset to training mode for next epoch
        model.train()
        
    save_model(model, tokenizer, model_name)
    
def main():
    # Initialize random seeds to ensure reproducible results
    set_seed(SEED)

    # Check for CUDA-capable GPU and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve model architecture and training hyperparameters from configuration
    # emb_dim: dimensionality of token embeddings and hidden states
    # num_layers: number of recurrent layers in the model
    # batch_size: mini-batch size
    # learning_rate: step size for optimizer updates
    # num_epochs: number of complete passes through the training dataset
    # context_size: maximum input sequence length

    # Initialize the tokenizer using Microsoft's Phi-3.5-mini model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    # Get the size of the vocabulary that the model needs to handle
    vocab_size = len(tokenizer)
    
    # get model
    model = get_model(vocab_size, EMB_DIM, NUM_LAYERS, tokenizer)
    
    # Initialize the loss function (Cross Entropy) for training
    # ignore_index=pad_token_id ensures that padding tokens don't contribute to the loss
    # This prevents the model from learning to predict padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Initialize the AdamW optimizer with specified learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Download the news dataset and create DataLoader objects for training and testing
    # DataLoaders handle batching and shuffling
    train_dataloader, test_dataloader = download_and_prepare_data(DATA_URL, BATCH_SIZE, tokenizer, CONTEXT_SIZE)
    train_model(
        model, 
        optimizer,
        criterion,
        NUM_EPOCHS, 
        train_dataloader,
        test_dataloader, 
        CONTEXT, 
        tokenizer,
        EVAL_INTERVAL,
        MAX_LENGTH,
        MODEL_NAME,
    )
    
if __name__ == "__main__":
    main()
