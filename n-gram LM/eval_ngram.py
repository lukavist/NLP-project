from collections import Counter

from ngram1 import NGramLanguageModel, evaluate_ngram_models


def detailed_analysis(training_text, test_text):
    """Perform detailed analysis of n-gram models trained on provided text."""
    print("DETAILED N-GRAM LANGUAGE MODEL ANALYSIS")
    print("=" * 60)

    # Train models
    unigram = NGramLanguageModel(n=1)
    bigram = NGramLanguageModel(n=2)
    trigram = NGramLanguageModel(n=3)

    print("Training models...")
    unigram.train(training_text)
    bigram.train(training_text)
    trigram.train(training_text)

    # Tokenize text for reference
    tokens = unigram.tokenize(training_text)
    unique_tokens = set(tokens)

    print(f"\nTraining Data Statistics:")
    print(f"- Total words: {len(tokens)}")
    print(f"- Unique words: {len(unique_tokens)}")
    print(f"- Top 10 most frequent words: {', '.join([word for word, _ in Counter(tokens).most_common(10)])}")

    # Analyze unigram probabilities
    print("\nUNIGRAM ANALYSIS")
    print("-" * 60)
    top_unigrams = unigram.most_common_ngrams(10)
    print("Top 10 unigrams with probabilities:")
    for word_tuple, count in top_unigrams:
        word = word_tuple[0]  # Extract word from tuple
        prob = unigram.probability(word)
        print(f"  '{word}': count={count}, P(word)={prob:.6f}")

    # Analyze bigram probabilities
    print("\nBIGRAM ANALYSIS")
    print("-" * 60)
    top_bigrams = bigram.most_common_ngrams(10)
    print("Top 10 bigrams with probabilities:")
    for word_pair, count in top_bigrams:
        prob = bigram.probability(word_pair)
        print(f"  '{word_pair[0]} {word_pair[1]}': count={count}, P({word_pair[1]}|{word_pair[0]})={prob:.6f}")

    # Analyze trigram probabilities
    print("\nTRIGRAM ANALYSIS")
    print("-" * 60)
    top_trigrams = trigram.most_common_ngrams(10)
    print("Top 10 trigrams with probabilities:")
    for word_triplet, count in top_trigrams:
        prob = trigram.probability(word_triplet)
        print(f"  '{word_triplet[0]} {word_triplet[1]} {word_triplet[2]}': count={count}, "
              f"P({word_triplet[2]}|{word_triplet[0]} {word_triplet[1]})={prob:.6f}")

    # Perplexity comparison
    print("\nPERPLEXITY COMPARISON")
    print("-" * 60)
    uni_perp = unigram.perplexity(test_text)
    bi_perp = bigram.perplexity(test_text)
    tri_perp = trigram.perplexity(test_text)

    print(f"Unigram perplexity: {uni_perp:.4f}")
    print(f"Bigram perplexity: {bi_perp:.4f}")
    print(f"Trigram perplexity: {tri_perp:.4f}")

    # Text generation comparison
    print("\nTEXT GENERATION COMPARISON")
    print("-" * 60)
    print(f"Unigram generated text (15 words): {unigram.generate_text(15)}")
    print(f"Bigram generated text (15 words): {bigram.generate_text(15)}")
    print(f"Trigram generated text (15 words): {trigram.generate_text(15)}")

    # Additional examples
    print("\nADDITIONAL PROBABILITY EXAMPLES")
    print("-" * 60)

    # Example 1: Same word, different models
    test_word = "language"
    if test_word in unigram.vocab:
        print(f"Probability of '{test_word}':")
        print(f"  Unigram P({test_word}) = {unigram.probability(test_word):.6f}")

        # Find contexts where this word appears
        for context in [c for c in bigram.context_counts.keys() if len(c) == 1]:
            if (context + (test_word,)) in bigram.ngram_counts:
                print(f"  Bigram P({test_word}|{context[0]}) = {bigram.probability(context + (test_word,)):.6f}")
                break

        for context in [c for c in trigram.context_counts.keys() if len(c) == 2]:
            if (context + (test_word,)) in trigram.ngram_counts:
                print(
                    f"  Trigram P({test_word}|{context[0]} {context[1]}) = {trigram.probability(context + (test_word,)):.6f}")
                break

    # Example 2: Effect of smoothing
    print("\nEffect of smoothing on unseen n-grams:")
    # Create an unseen unigram, bigram, and trigram
    unseen_word = "cybersecurity"  # Hopefully not in the training text
    if unseen_word not in unigram.vocab:
        print(f"  Unigram P({unseen_word}) with smoothing = {unigram.probability(unseen_word):.6f}")
        print(f"  Unigram P({unseen_word}) without smoothing = {unigram.probability(unseen_word, smoothing=False):.6f}")

        common_word = top_unigrams[0][0][0]  # Most common word
        unseen_bigram = (common_word, unseen_word)
        print(f"  Bigram P({unseen_word}|{common_word}) with smoothing = {bigram.probability(unseen_bigram):.6f}")
        print(
            f"  Bigram P({unseen_word}|{common_word}) without smoothing = {bigram.probability(unseen_bigram, smoothing=False):.6f}")


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

    # Run detailed analysis
    detailed_analysis(training_data, test_data)

    print("\n" + "=" * 60)
    print("Standard evaluation of all models:")
    evaluate_ngram_models(training_data, test_data)