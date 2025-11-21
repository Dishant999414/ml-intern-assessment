import random
import re
from collections import defaultdict, Counter


class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Stores trigram counts: (w1, w2) → Counter(w3)
        self.trigram_counts = defaultdict(Counter)

        # Vocabulary of words (for safety in generation)
        self.vocab = set()

        # To store list of all words for simple token lookup
        self.tokens = []

    def _clean_and_tokenize(self, text):
        """
        Lowercase + remove punctuations + split tokens
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = text.split()
        return tokens

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        tokens = self._clean_and_tokenize(text)
        self.tokens = tokens

        # Add start and end padding
        padded = ["<s>", "<s>"] + tokens + ["</s>"]

        # Build trigram counts
        for i in range(len(padded) - 2):
            w1, w2, w3 = padded[i], padded[i+1], padded[i+2]
            self.trigram_counts[(w1, w2)][w3] += 1
            self.vocab.add(w3)

    def _get_next_word(self, w1, w2):
        """
        Randomly chooses next word based on trigram counts.
        """
        possible = self.trigram_counts.get((w1, w2), None)

        if not possible:
            # fallback if context not found
            return random.choice(list(self.vocab))

        # Weighted random choice
        words = list(possible.keys())
        weights = list(possible.values())
        return random.choices(words, weights=weights, k=1)[0]

    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): Maximum words in output

        Returns:
            str: Generated text.
        """
        w1, w2 = "<s>", "<s>"
        output = []

        for _ in range(max_length):
            w3 = self._get_next_word(w1, w2)

            if w3 == "</s>":
                break

            output.append(w3)
            w1, w2 = w2, w3

        return " ".join(output)
