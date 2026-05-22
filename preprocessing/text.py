from __future__ import annotations

import re
import string
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


@dataclass
class TextPreprocessConfig:
    lowercase: bool = True
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    stemming: bool = True
    lemmatization: bool = True


class TrustifyTextPreprocessor:
    """Paper-aligned cleaning: stopwords, punctuation, stemming, lemmatization."""

    def __init__(self, config: TextPreprocessConfig):
        self.config = config
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        try:
            self._stopwords = set(stopwords.words("english"))
        except LookupError as exc:
            raise RuntimeError(
                "NLTK stopwords are required. Run: python -m nltk.downloader stopwords wordnet omw-1.4 punkt"
            ) from exc

    def clean(self, text: str) -> list[str]:
        text = "" if text is None else str(text)
        if self.config.lowercase:
            text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
        tokens = text.split()
        if self.config.remove_stopwords:
            tokens = [tok for tok in tokens if tok not in self._stopwords]
        if self.config.stemming:
            tokens = [self._stemmer.stem(tok) for tok in tokens]
        if self.config.lemmatization:
            tokens = [self._lemmatizer.lemmatize(tok) for tok in tokens]
        return tokens


def build_text_preprocessor(config: dict) -> TrustifyTextPreprocessor:
    return TrustifyTextPreprocessor(TextPreprocessConfig(**{
        "lowercase": config.get("lowercase", True),
        "remove_stopwords": config.get("remove_stopwords", True),
        "remove_punctuation": config.get("remove_punctuation", True),
        "stemming": config.get("stemming", True),
        "lemmatization": config.get("lemmatization", True),
    }))

