import nltk
from nltk import pos_tag, RegexpParser
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from nltk.tree import Tree

'''
NP: noun phrase, has determiner <DT>(the, an), <JJ> adjective, <NN> nouns 
(.* means it covers plural nouns etc; (+ means it can consist more than one)

VP: <VB.*> covers all types of verbs (past tense, base form); then verb is followed by one or more 
NP - noun phrase, PP - prepositional phrase, or S(clause); $ - means that VP must end with these elements

PP: prepositional phrase, consists of preposition <IN> followed by <NP> noun phrase

S(clause): consists of <NP> - noun phrase followed by <VP> - verb phrase
'''


class TextParser:
    def __init__(self):
        # Define a grammar
        self.grammar = r"""
            NP: {<DT|JJ|NN.*>+}          # Noun phrase
            VP: {<VB.*><NP|PP|CLAUSE>+$} # Verb phrase
            PP: {<IN><NP>}               # Prepositional phrase
            CLAUSE: {<NP><VP>}           # Simple clause
        """
        # Create a chunk parser
        self.parser = RegexpParser(self.grammar)

    def tokenize(self, text):
        sentences = sent_tokenize(text)
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        return tokenized_sentences

    def tag_parts_of_speech(self, tokenized_sentences):
        tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]
        return tagged_sentences

    def parse_chunks(self, tagged_sentences):
        chunked_sentences = [self.parser.parse(tagged_sentence) for tagged_sentence in tagged_sentences]
        return chunked_sentences

    def visualize_parse_tree(self, tree):
        if isinstance(tree, nltk.tree.Tree):
            tree.draw()
        else:
            print("Input is not a valid Tree object.")

    def analyze_text(self, text):
        tokenized = self.tokenize(text)
        tagged = self.tag_parts_of_speech(tokenized)
        chunked = self.parse_chunks(tagged)

        results = {
            'tokenized': tokenized,
            'tagged': tagged,
            'chunked': chunked
        }

        return results


# Example usage and demonstration
def demo_parser():
    parser = TextParser()

    # Example sentences
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "I enjoy reading books about artificial intelligence.",
        "She walked to the store and bought some groceries."
    ]

    print("TEXT PARSER DEMONSTRATION\n")

    for i, example in enumerate(examples):
        print(f"Example {i + 1}: \"{example}\"")

        # Analyze the text
        results = parser.analyze_text(example)

        # Display tokenized words
        print("\nTokenized words:")
        for sentence in results['tokenized']:
            print(sentence)

        # Display POS tags
        print("\nParts of speech:")
        for sentence in results['tagged']:
            print(sentence)

        # Display chunked structure in text form
        print("\nChunked structure:")
        for tree in results['chunked']:
            print(tree)

        # Print a structured representation
        # print("\nStructured analysis:")
        # for tree in results['chunked']:
        #    for subtree in tree.subtrees():
        #        if subtree.label() != 'S':  # Skip the sentence level
        #            print(f"{subtree.label()}: {' '.join(word for word, tag in subtree.leaves())}")

        print("\n" + "-" * 50 + "\n")

    # print("To visualize parse trees graphically, use:")
    # print("parser.visualize_parse_tree(results['chunked'][0])")


if __name__ == "__main__":
    demo_parser()





