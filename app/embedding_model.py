import spacy


class EmbeddingModel:
    def __init__(self):
        # Load spaCy's large English model
        self.nlp = spacy.load("en_core_web_lg")

    def calculate_embedding(self, input_word):
        # Convert the input word into a spaCy document
        doc = self.nlp(input_word)

        # Return the word embedding as a Python list
        return doc.vector.tolist()

    def calculate_similarity(self, word1, word2):
        # Process both words
        doc1 = self.nlp(word1)
        doc2 = self.nlp(word2)

        # Calculate and return cosine similarity
        return float(doc1.similarity(doc2))