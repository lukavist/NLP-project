from nltk import pos_tag, RegexpParser
from nltk.tokenize import word_tokenize, sent_tokenize

# Plan
# 1. Create simple grammar
# 2. Tokenize sentence into words (list of words)
# 3. Do POS tagging
# 4. Identify phrase structures with chunk grammar


'''
NP: noun phrase, has determiner <DT>(the, an), <JJ> adjective, <NN> nouns 
(.* means it covers plural nouns etc; (+ means it can consist more than one)

VP: <VB.*> covers all types of verbs (past tense, base form); then verb is followed by one or more 
NP - noun phrase, PP - prepositional phrase, or S(clause); $ - means that VP must end with these elements

PP: prepositional phrase, consists of preposition <IN> followed by <NP> noun phrase

S(clause): consists of <NP> - noun phrase followed by <VP> - verb phrase
'''


grammar = r"""
    NP: {<DT>?<JJ>*<NN.*>+} # noun phrase
    VP: {<VB.*><NP|PP|S>+$} # verb phrase
    PP: {<IN><NP>}          # prepositional phrase
    S: {<NP><VP>}           # clause
"""

parser = RegexpParser(grammar)


# tokenize sentence into list of words
def tokenize(text):
    tokenized_sentence = []

    sentences = sent_tokenize(text)

    for sentence in sentences:

        tokenized_sentence.append(word_tokenize(sentence))

    return tokenized_sentence


# POS tag tokenized sentences
def pos_tagger(tokenized_sentence):
    tagged_sentence = []

    for sentence in tokenized_sentence:

        tagged_sentence.append(pos_tag(sentence))

    return tagged_sentence


# create hierarchical chunks
def chunks_parser(tagged_sentence):
    chunks_sentences = []

    for sentence in tagged_sentence:

        chunks_sentences.append(parser.parse(sentence))

    return chunks_sentences


def analyze_text(text):
    tokenized = tokenize(text)
    tagged = pos_tagger(tokenized)
    chunked = chunks_parser(tagged)

    result = {
        "tokenized": tokenized,
        "tagged": tagged,
        "chunked": chunked
    }

    return result

def text_parser():

    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "I enjoy reading books about artificial intelligence.",
        "She walked to the store and bought some groceries."
    ]

    print("DEMO PRESENTATION OF TEXT PARSER")

    for i, sentence in enumerate(examples):
        print(f"\nEXAMPLE {i+1}: \n{sentence}")

        results = analyze_text(sentence)

        print("\nTOKENIZED")

        for result in results["tokenized"]:
            print(result)

        print("\nPOS TAGGER")
        for result in results["tagged"]:
            print(result)

        print("\nCHUNKED")
        for result in results["chunked"]:
            print(result)



if __name__ == '__main__':
    text_parser()
