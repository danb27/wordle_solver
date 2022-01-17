from collections import Counter

from english_words import english_words_lower_alpha_set
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import CorrectOrCloseWords, EligibleWords, ScoredStrings, WrongWords, WORDS_TO_ADD, WORDS_TO_REMOVE


class WorldleSolver:
    def __init__(self, word_length: int = 5, use_tfidf: bool = True):
        """
        WordleSolver uses an english lexicon to make intelligent guesses when playing Wordle.

        :param word_length: How many characters does the target word have? The classic game uses 5, but can be adjusted
        here if needed.
        :param use_tfidf: Whether to use a tfidf of the letters, or just simple term frequency
        """
        self.word_length = word_length
        self.use_tfidf = use_tfidf
        # for some reason this dataset has words with periods
        # limit to words with only 5 alpha characters
        words_raw = {word.replace('.', '') for word in english_words_lower_alpha_set}

        # add words that are not in english_words
        # but are in your version of the game
        words_raw.update(WORDS_TO_ADD)

        # keep the word and a list of the characters for easy indexing later
        self.words = {word: list(word) for word in words_raw if
                      len(word) == self.word_length and word not in WORDS_TO_REMOVE}

        if self.use_tfidf:
            self.tfidf = self._calculate_letter_tfidf()
        else:
            self.letter_frequencies = self._calculate_letter_freq()

        # require the first guess to have all unique characters.
        # this will help us get the most information from the first guess.
        self._print_guess(self.words, True, True)

    def _print_guess(self, words: EligibleWords, force_unique_letters: bool = False, first: bool = False):
        """
        Given some eligible words, score them and print what the user's guess should be.
        :param words: a dictionary {'word': ['w', 'o', 'r', 'd'], ...} of eligible words (words that satisfy all the
        known clues so far
        :param force_unique_letters: whether or not to force the guess to have no repeated characters
        :param first: whether this is the first guess or not (only changes the print statement)
        :return:
        """
        print(f"{'first' if first else 'next'} guess: {self._order_words(words, force_unique_letters)}")

    def _calculate_letter_freq(self) -> ScoredStrings:
        """
        Calculate how common every letter is (should all add up to 1)
        :return: dict with letters as keys and a float between 0 and 1 as the value
        """
        total_chars = len(self.words) * self.word_length
        frequencies = Counter()
        for chars in self.words.values():
            frequencies.update(chars)

        frequencies = {k: v / total_chars for k, v in dict(frequencies).items() if k.isalpha()}
        return dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))

    def _calculate_letter_tfidf(self) -> TfidfVectorizer:
        """
        Fit a tfidf on the characters in your documents (words)
        :return: the fit tfidf to be used when making predictions/guesses
        """
        tfidf = TfidfVectorizer(analyzer='char')
        tfidf.fit(self.words.keys())
        return tfidf

    def _order_words(self, words: EligibleWords, force_unique_letters: bool = False,  n: int = 3) -> ScoredStrings:
        """
        Score every eligible word and return the top n in order by score. Will use TFIDF if self.use_tfidf, if not,
        will use just Term Frequency.
        :param words: a dictionary of words that are eligible to be the target (satisfy all clues given). format is
        {'word': ['w', 'o', 'r', 'd']}
        :param force_unique_letters: whether or not to force all letters in the prediciton/guess to be unique
        :param n: how many of the top predictions to return
        :return: a dict of top n words and their scores
        """
        if force_unique_letters:
            words = {word: chars for word, chars in words.items() if len(set(word)) == len(chars)}
        if self.use_tfidf:
            scores = {word: self.tfidf.transform([word]).sum() for word in words}
        else:
            scores = {word: sum(self.letter_frequencies[char] for char in chars) for word, chars in words.items()}
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n])

    def provide_clues(self, correct: CorrectOrCloseWords, close:  CorrectOrCloseWords,
                      wrong: WrongWords):
        """
        Provide clues to the solver so that it can print out the best recommendations. Make sure to provide clues
        from all guesses, not just the latest one (if you have guessed more than once).
        :param correct: list of tuples ('{char}': index) representing green tiles
        :param close: list of tuples ('{char}': index) representing yellow tiles
        :param wrong: list of letters representing gray tiles
        :return:
        """
        words_to_keep = {}
        for word, chars in self.words.items():
            if any(chars[idx] != char for char, idx in correct):
                continue
            if any((chars[idx] == char or char not in chars) for char, idx in close):
                continue
            if any(char in chars for char in wrong):
                continue
            words_to_keep[word] = chars

        self._print_guess(words_to_keep)
