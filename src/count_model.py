
# Import required libraries
import re         # For regular expressions (text tokenization)
import requests   # For downloading the corpus
import gzip       # For decompressing the downloaded corpus
import io         # For handling byte streams
import math       # For mathematical operations (log, exp)
import random     # For random number generation
from collections import defaultdict  # For efficient dictionary operations
import pickle, os # For saving and loading the model
from loggers import strim_logger


def set_seed(seed):
    """
    Sets random seeds for reproducibility.

    Args:
        seed (int): Seed value for the random number generator
    """
    random.seed(seed)
    

def download_corpus(url):
    """
    Downloads and decompresses a gzipped corpus file from the given URL.

    Args:
        url (str): URL of the gzipped corpus file

    Returns:
        str: Decoded text content of the corpus

    Raises:
        HTTPError: If the download fails
    """
    strim_logger.info(f"Downloading corpus from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for bad HTTP responses

    strim_logger.info("Decompressing and reading the corpus...")
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        corpus = f.read().decode('utf-8')

    strim_logger.info(f"Corpus size: {len(corpus)} characters")
    return corpus


class CountLanguageModel:
    """
    Implements an n-gram language model using count-based probability estimation.
    Supports variable context lengths up to n-grams.
    """
    def __init__(self, n):
        """
        Initialize the model with maximum n-gram length.

        Args:
            n (int): Maximum length of n-grams to use
        """
        self.n = n  # Maximum n-gram length
        self.ngram_counts = [{} for _ in range(n)]  # List of dictionaries for each n-gram length
        self.total_unigrams = 0  # Total number of tokens in training data

    def predict_next_token(self, context):
        """
        Predicts the most likely next token given a context.
        Uses backoff strategy: tries largest n-gram first, then backs off to smaller n-grams.

        Args:
            context (list): List of tokens providing context for prediction

        Returns:
            str: Most likely next token, or None if no prediction can be made
        """
        for n in range(self.n, 1, -1):  # Start with largest n-gram, back off to smaller ones
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])  # Get the relevant context for this n-gram
                counts = self.ngram_counts[n - 1].get(context_n)
                if counts:
                    return max(counts.items(), key=lambda x: x[1])[0]  # Return most frequent token
        # Backoff to unigram if no larger context matches
        unigram_counts = self.ngram_counts[0].get(())
        if unigram_counts:
            return max(unigram_counts.items(), key=lambda x: x[1])[0]
        return None

    def get_probability(self, token, context):
        for n in range(self.n, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])
                counts = self.ngram_counts[n - 1].get(context_n)
                if counts:
                    total = sum(counts.values())
                    count = counts.get(token, 0)
                    if count > 0:
                        return count / total
        unigram_counts = self.ngram_counts[0].get(())
        count = unigram_counts.get(token, 0)
        V = len(unigram_counts)
        return (count + 1) / (self.total_unigrams + V)
    
    
def train(model, tokens):
    """
    Trains the language model by counting n-grams in the training data.

    Args:
        model (CountLanguageModel): Model to train
        tokens (list): List of tokens from the training corpus
    """
    # Train models for each n-gram size from 1 to n
    for n in range(1, model.n + 1):
        counts = model.ngram_counts[n - 1]
        # Slide a window of size n over the corpus
        for i in range(len(tokens) - n + 1):
            # Split into context (n-1 tokens) and next token
            context = tuple(tokens[i:i + n - 1])
            next_token = tokens[i + n - 1]

            # Initialize counts dictionary for this context if needed
            if context not in counts:
                counts[context] = defaultdict(int)

            # Increment count for this context-token pair
            counts[context][next_token] = counts[context][next_token] + 1

    # Store total number of tokens for unigram probability calculations
    model.total_unigrams = len(tokens)
    
    
def compute_perplexity(model, tokens, context_size):
    """
    Computes perplexity of the model on given tokens.

    Args:
        model (CountLanguageModel): Trained language model
        tokens (list): List of tokens to evaluate on
        context_size (int): Maximum context size to consider

    Returns:
        float: Perplexity score (lower is better)
    """
    # Handle empty token list
    if not tokens:
        return float('inf')

    # Initialize log likelihood accumulator
    total_log_likelihood = 0
    num_tokens = len(tokens)

    # Calculate probability for each token given its context
    for i in range(num_tokens):
        # Get appropriate context window, handling start of sequence
        context_start = max(0, i - context_size)
        context = tuple(tokens[context_start:i])
        token = tokens[i]

        # Get probability of this token given its context
        probability = model.get_probability(token, context)

        # Add log probability to total (using log for numerical stability)
        total_log_likelihood += math.log(probability)

    # Calculate average log likelihood
    average_log_likelihood = total_log_likelihood / num_tokens

    # Convert to perplexity: exp(-average_log_likelihood)
    # Lower perplexity indicates better model performance
    perplexity = math.exp(-average_log_likelihood)
    return perplexity

