import string
import math
import random
from collections import Counter, defaultdict


class NGramLanguageModel:
    def __init__(self, n=1):

        self.n = n
        self.ngram_counts = Counter()  # Counts of complete n-grams
        self.context_counts = Counter()  # Counts of (n-1)-grams (contexts)
        self.vocab = set()  # Set of all unique words
        self.total_words = 0  # Total number of words in training data

    def tokenize(self, text):

        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text = text.lower().translate(translator)

        # Split on whitespace
        tokens = text.split()
        return tokens

    def train(self, text):

        tokens = self.tokenize(text)
        self.total_words = len(tokens)
        self.vocab.update(tokens)

        # Count n-grams
        for i in range(len(tokens) - self.n + 1):
            # Extract the current n-gram
            ngram = tuple(tokens[i:i + self.n])
            self.ngram_counts[ngram] += 1

            # For n > 1, count the context
            if self.n > 1:
                context = ngram[:-1]  # All except the last word
                self.context_counts[context] += 1
            else:
                # For unigrams just track total words
                self.context_counts[()] = self.total_words

    def probability(self, ngram, smoothing=True):

        if isinstance(ngram, str):
            ngram = (ngram,)

        # Get counts
        ngram_count = self.ngram_counts[ngram]

        if self.n == 1:
            # Unigram: P(word) = count(word) / total_words
            denominator = self.total_words
        else:
            # N-gram: P(word_n | word_1, ..., word_{n-1}) = count(n-gram) / count(context)
            context = ngram[:-1]
            denominator = self.context_counts[context]

            # If context never seen, back off to total words
            if denominator == 0:
                denominator = self.total_words

        # Apply smoothing
        if smoothing:
            # Add Laplace smoothing
            return (ngram_count + 1) / (denominator + len(self.vocab))
        else:
            # No smoothing
            return ngram_count / denominator if denominator > 0 else 0

    def perplexity(self, test_text):

        tokens = self.tokenize(test_text)

        if len(tokens) < self.n:
            return float('inf')  # Not enough tokens for this n-gram size

        log_prob_sum = 0
        count = 0

        # Calculate probability for each possible n-gram in test text
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            prob = self.probability(ngram)

            # Avoid log(0) by enforcing a minimum probability
            if prob > 0:
                log_prob_sum += math.log2(prob)
                count += 1

        # If no valid probabilities were found
        if count == 0:
            return float('inf')

        # Calculate perplexity: 2^(-average log probability)
        return 2 ** (-log_prob_sum / count)

    def generate_text(self, length=20):

        if not self.ngram_counts:
            return "Model not trained yet."

        # Start with a random n-gram prefix if n > 1
        if self.n > 1:
            # Get all possible contexts (prefixes)
            contexts = list(self.context_counts.keys())
            if not contexts:
                return "No valid contexts found."

            # Choose a random context to start with
            current = random.choice(contexts)
            result = list(current)
        else:
            # For unigrams start with empty result
            result = []
            current = ()

        # Generate words one by one
        for _ in range(length - len(result)):
            # Find all words that follow the current context
            possible_words = [ngram[-1] for ngram in self.ngram_counts.keys()
                              if ngram[:-1] == current]

            # If no continuations found pick a random word from vocabulary
            if not possible_words:
                if not self.vocab:
                    break
                next_word = random.choice(list(self.vocab))
            else:
                # Sample according to n-gram probabilities
                probs = [self.probability(current + (word,)) for word in possible_words]
                next_word = random.choices(possible_words, weights=probs)[0]

            # Add generated word to result
            result.append(next_word)

            # Update context for next iteration
            if self.n > 1:
                current = tuple(result[-self.n + 1:])

        return ' '.join(result)

    def most_common_ngrams(self, limit=10):
        return self.ngram_counts.most_common(limit)


# Function to evaluate all three n-gram models
def evaluate_ngram_models(training_text, test_text):

    results = {}

    # Process each n-gram size
    for n in [1, 2, 3]:
        print(f"\nTraining {n}-gram model...")
        model = NGramLanguageModel(n)
        model.train(training_text)

        # Vocabulary statistics
        vocab_size = len(model.vocab)
        print(f"Vocabulary size: {vocab_size} words")

        # Most common n-grams
        print(f"Top 5 most common {n}-grams:")
        for ngram, count in model.most_common_ngrams(5):
            print(f"  {' '.join(ngram)}: {count}")

        # Calculate perplexity
        perplexity = model.perplexity(test_text)
        print(f"Perplexity on test text: {perplexity:.4f}")

        # Generate sample text
        generated_text = model.generate_text(15)
        print(f"Sample generated text: {generated_text}")

        # Store results
        results[n] = {
            'model': model,
            'vocab_size': vocab_size,
            'perplexity': perplexity,
            'generated_text': generated_text
        }

    return results


# Main function to run the program
if __name__ == "__main__":
    # Sample training data
    training_data = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
    It focuses on the interactions between computers and human language.
    The goal of NLP is to enable computers to understand, interpret, and generate human language in a valuable way.
    Modern NLP algorithms are based on machine learning techniques.
    Processing of natural language is challenging because human language is complex and often ambiguous.
    Language understanding involves knowledge about words, grammar, culture, and common sense.
    Deep learning models have revolutionized natural language processing in recent years.
    Transformers and other neural network architectures have achieved impressive results across many NLP tasks.
    """

    # Sample test data
    test_data = """
    Natural language processing systems help computers understand and respond to human language.
    These systems use machine learning algorithms to analyze text and speech.
    Modern approaches to NLP rely heavily on deep learning techniques.
    """

    print("N-gram Language Model Evaluation")
    print("=" * 50)

    results = evaluate_ngram_models(training_data, test_data)

    print("\nSummary of Results:")
    print("-" * 50)
    for n in [1, 2, 3]:
        print(f"{n}-gram model:")
        print(f"  Perplexity: {results[n]['perplexity']:.4f}")
        print(f"  Vocabulary size: {results[n]['vocab_size']}")
        print(f"  Sample text: {results[n]['generated_text']}")