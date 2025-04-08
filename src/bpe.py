# Import required libraries
import os  # For file operations and path handling
import urllib.request  # For downloading files
import tarfile  # For extracting tar files
import pickle  # For saving/loading tokenizer
import re  # For regex in merge operations
import time  # For timing operations
from collections import defaultdict  # For counting tokens and pairs

from loggers import strim_logger



def download_file(url, filename):
    """
    Downloads a file from a URL if it doesn't exist locally.
    Prevents redundant downloads by checking file existence.

    Args:
        url (str): URL to download the file from
        filename (str): Local path to save the downloaded file

    Returns:
        None: Prints status messages about download progress
    """
    # Check if file already exists to avoid re-downloading
    if not os.path.exists(filename):
        strim_logger.info(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filename)   # Copy a network object denoted by a URL to a local file;  Return a tuple (filename, headers)
        strim_logger.info("Download completed.")
    else:
        strim_logger.info(f"{filename} already downloaded.")


def is_within_directory(directory, target):
    """
    Security check to prevent path traversal attacks by verifying target path.
    Ensures extracted files remain within the intended directory.

    Args:
        directory (str): Base directory path to check against
        target (str): Target path to validate

    Returns:
        bool: True if target is within directory, False otherwise
    """
    # Convert both paths to absolute form for comparison
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    # Get common prefix to check containment
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory


def safe_extract_tar(tar_file, required_files):
    """
    Safely extracts specific files from a tar archive with security checks.
    Prevents path traversal attacks and extracts only required files.

    Args:
        tar_file (str): Path to the tar archive file
        required_files (list): List of filenames to extract

    Returns:
        None: Extracts files and prints progress

    Raises:
        Exception: If path traversal attempt is detected
    """
    with tarfile.open(tar_file, "r:gz") as tar:
        # Perform security check on all archive members
        for member in tar.getmembers():
            if not is_within_directory('.', member.name):
                raise Exception("Attempted Path Traversal in Tar File")

        # Extract only the specified files
        for member in tar.getmembers():
            if any(member.name.endswith(file) for file in required_files):
                # Remove path prefix for safety
                member.name = os.path.basename(member.name)
                tar.extract(member, '.')
                strim_logger.info(f"Extracted {member.name}")
                
                
def create_word_generator(filepath):
    """
    Creates a generator that yields words from a text file one at a time.
    Memory efficient way to process large text files.

    Args:
        filepath (str): Path to text file to read

    Returns:
        generator: Yields individual words from the file
    """
    def generator():
        with open(filepath, 'r') as f:
            for line in f:
                for word in line.split():
                    yield word
    return generator()


def download_and_prepare_data(url):
    """
    Downloads, extracts, and prepares dataset for training.
    Handles both downloading and extraction with security checks.

    Args:
        url (str): URL of the dataset to download

    Returns:
        generator: Word generator for the training data
    """
    required_files = ["train.txt", "test.txt"]
    filename = os.path.basename(url)

    # Download dataset if needed
    download_file(url, filename)

    # Extract required files if they don't exist
    if not all(os.path.exists(file) for file in required_files):
        strim_logger.info("Extracting files...")
        safe_extract_tar(filename, required_files)
        strim_logger.info("Extraction completed.")
    else:
        strim_logger.info("'train.txt' and 'test.txt' already extracted.")

    # Create and return word generator
    return create_word_generator("train.txt")



def initialize_vocabulary(corpus):
    """
    Creates initial vocabulary from corpus by splitting words into characters.
    Adds word boundary marker '_' and tracks unique characters.

    Args:
        corpus (iterable): Iterator or list of words to process

    Returns:
        tuple: (vocabulary dict mapping tokenized words to counts,
               set of unique characters in corpus)
    """
    # Track word counts and unique characters
    vocabulary = defaultdict(int)
    charset = set()

    for word in corpus:
        # Add word boundary marker and split into characters
        word_with_marker = '_' + word
        characters = list(word_with_marker)
        # Update set of unique characters
        charset.update(characters)
        # Create space-separated string of characters
        tokenized_word = " ".join(characters)
        # Increment count for this tokenized word
        vocabulary[tokenized_word] += 1

    return vocabulary, charset


def get_pair_counts(vocabulary):
    """
    Counts frequencies of adjacent symbol pairs in the vocabulary.
    Used to identify most common pairs for merging.

    Args:
        vocabulary (dict): Dictionary mapping tokenized words to their counts

    Returns:
        defaultdict: Maps token pairs to their frequency counts
    """
    pair_counts = defaultdict(int)
    for tokenized_word, count in vocabulary.items():
        # Split word into tokens
        tokens = tokenized_word.split()
        # Count adjacent pairs weighted by word frequency
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
    return pair_counts



def merge_pair(vocab, pair):
    """
    Merges all occurrences of a specific symbol pair in the vocabulary.
    Uses regex for accurate token boundary matching.

    Args:
        vocab (dict): Current vocabulary dictionary
        pair (tuple): Pair of tokens to merge

    Returns:
        dict: New vocabulary with specified pair merged
    """
    new_vocab = {}
    # Create regex pattern for matching the pair
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

    # Apply merge to all words in vocabulary
    for tokenized_word, count in vocab.items():
        new_tokenized_word = pattern.sub("".join(pair), tokenized_word)
        new_vocab[new_tokenized_word] = count
    return new_vocab


def byte_pair_encoding(corpus, vocab_size):
    """
    Implements the BPE algorithm to learn a subword vocabulary.
    Iteratively merges most frequent character pairs until target vocabulary size is reached.

    Args:
        corpus (iterable): Iterator or list of words to learn BPE from
        vocab_size (int): Target vocabulary size to stop merging at

    Returns:
        tuple: (final vocabulary dict, list of merge operations,
               set of base characters, set of all tokens)
    """
    # Initialize vocabulary with character-level tokens
    vocab, charset = initialize_vocabulary(corpus)
    merges = []
    tokens = set(charset)

    # Keep merging pairs until we reach target vocab size
    while len(tokens) < vocab_size:
        # Get counts of all adjacent token pairs
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break

        # Find and record the most frequent pair
        most_frequent_pair = max(pair_counts, key=pair_counts.get)
        merges.append(most_frequent_pair)

        # Update vocabulary by merging the most frequent pair
        vocab = merge_pair(vocab, most_frequent_pair)

        # Add the new merged token to our token set
        new_token = "".join(most_frequent_pair)
        tokens.add(new_token)

    return vocab, merges, charset, tokens


def tokenize_word(word, merges, charset, unk_token="<UNK>"):
    """
    Tokenizes a single word using learned BPE merges.
    Handles unknown characters with UNK token.

    Args:
        word (str): Word to tokenize
        merges (list): List of learned merge operations
        charset (set): Set of known characters
        unk_token (str): Token to use for unknown characters

    Returns:
        list: List of tokens for the word
    """
    # Add word boundary marker and convert to characters
    word = '_' + word
    tokens = [char if char in charset else unk_token for char in word]

    # Apply merges in order
    for left, right in merges:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i:i+2] == [left, right]:
                tokens[i:i+2] = [left + right]
            else:
                i += 1
    return tokens


def build_merge_map(merges):
    """
    Creates a mapping from token pairs to their merged forms.
    Preserves merge order for consistent tokenization.

    Args:
        merges (list): List of merge operations

    Returns:
        dict: Maps token pairs to (merged_token, merge_priority) tuples
    """
    merge_map = {}
    # Build map with merge priorities
    for i, (left, right) in enumerate(merges):
        merged_token = left + right
        merge_map[(left, right)] = (merged_token, i)
    return merge_map



def tokenize_word_fast(word, merge_map, vocabulary, charset, unk_token="<UNK>"):
    """
    Optimized tokenization function using pre-computed merge map.
    Produces identical results to original algorithm but faster.

    Args:
        word (str): Word to tokenize
        merge_map (dict): Mapping of token pairs to merged forms
        vocabulary (dict): Current vocabulary dictionary
        charset (set): Set of known characters
        unk_token (str): Token to use for unknown characters

    Returns:
        list: List of tokens for the word
    """
    # Check if word exists in vocabulary as-is
    word_with_prefix = '_' + word
    if word_with_prefix in vocabulary:
        return [word_with_prefix]

    # Initialize with characters, replacing unknown ones
    tokens = [char if char in charset else unk_token for char in word_with_prefix]

    # Keep merging until no more merges possible
    while True:
        # Find all possible merge operations
        pairs_with_positions = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_map:
                merged_token, merge_priority = merge_map[pair]
                pairs_with_positions.append((i, pair, merged_token, merge_priority))

        # Exit if no more merges possible
        if not pairs_with_positions:
            break

        # Sort by merge priority and position for consistency
        pairs_with_positions.sort(key=lambda x: (x[3], x[0]))

        # Apply first valid merge
        pos, pair, merged_token, _ = pairs_with_positions[0]
        tokens[pos:pos+2] = [merged_token]

    return tokens

def save_tokenizer(merges, charset, tokens, filename="tokenizer.pkl"):
    """
    Saves tokenizer state to a pickle file for later use.

    Args:
        merges (list): List of merge operations
        charset (set): Set of known characters
        tokens (set): Set of all tokens
        filename (str): Path to save tokenizer state

    Returns:
        None: Saves tokenizer to disk
    """
    with open(filename, "wb") as f:
        pickle.dump({
            "merges": merges,
            "charset": charset,
            "tokens": tokens
        }, f)

def load_tokenizer(filename="tokenizer.pkl"):
    """
    Loads tokenizer state from a pickle file.

    Args:
        filename (str): Path to saved tokenizer state

    Returns:
        dict: Dictionary containing tokenizer components
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
    
    
    
# Main function for downloading, training BPE, saving, and loading tokenizer
if __name__ == "__main__":
    # Configuration parameters
    vocab_size = 300  # Target vocabulary size
    max_corpus_size = 50_000  # Maximum number of words to process
    data_url = "https://www.thelmbook.com/data/news"  # Dataset source

    # Download and prepare training data
    word_gen = download_and_prepare_data(data_url)

    # Collect corpus up to maximum size
    word_list = []
    for word in word_gen:
        word_list.append(word)
        if len(word_list) >= max_corpus_size:
            break

    # Train BPE tokenizer
    strim_logger.info("Training BPE tokenizer...")
    vocab, merges, charset, tokens = byte_pair_encoding(word_list, vocab_size)

    # Save trained tokenizer
    strim_logger.info("Saving the tokenizer...")
    save_tokenizer(merges, charset, tokens)