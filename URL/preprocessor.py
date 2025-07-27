import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse
from collections import Counter
import pickle
import logging

logger = logging.getLogger(__name__)


class URLPreprocessor:
    """URL preprocessing with character-level encoding"""

    DEFAULT_SPECIAL_CHARS = [
        " ",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
    ]

    def __init__(
        self, max_len: int = 500, num_chars: int = 150, min_char_freq: int = 10
    ):
        self.char_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_char = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        self.max_len = max_len
        self.num_chars = num_chars
        self.min_char_freq = min_char_freq
        self.is_fitted = False

    def create_char_dict(self, urls: list) -> None:
        char_counts = Counter()
        for url in urls:
            if self.validate_url(url):
                char_counts.update(url.lower())

        chars = {
            char for char, count in char_counts.items() if count >= self.min_char_freq
        }
        chars.update(self.DEFAULT_SPECIAL_CHARS)

        for idx, char in enumerate(sorted(chars), start=2):
            self.char_index[char] = idx
            self.index_char[idx] = char

        self.vocab_size = min(len(self.char_index), self.num_chars)
        self.is_fitted = True
        logger.info(f"Created vocabulary with {self.vocab_size} unique characters")

    @staticmethod
    def validate_url(url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            logger.warning(f"Invalid URL: {url}")
            return False

    def preprocess_urls(self, urls: list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call create_char_dict() first.")

        processed_urls = []
        for url in urls:
            if self.validate_url(url):
                seq = [
                    self.char_index.get(char.lower(), self.char_index["<UNK>"])
                    for char in url
                ]
                processed_urls.append(seq)
            else:
                processed_urls.append([self.char_index["<PAD>"]] * self.max_len)

        return pad_sequences(
            processed_urls,
            maxlen=self.max_len,
            padding="post",
            truncating="post",
            value=self.char_index["<PAD>"],
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "char_index": self.char_index,
                    "index_char": self.index_char,
                    "vocab_size": self.vocab_size,
                    "max_len": self.max_len,
                    "num_chars": self.num_chars,
                    "min_char_freq": self.min_char_freq,
                    "is_fitted": self.is_fitted,
                },
                f,
            )
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "URLPreprocessor":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        preprocessor = cls(
            data["max_len"], data["num_chars"], data.get("min_char_freq", 10)
        )
        preprocessor.char_index = data["char_index"]
        preprocessor.index_char = data["index_char"]
        preprocessor.vocab_size = data["vocab_size"]
        preprocessor.is_fitted = data["is_fitted"]
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
