"""Module that contains Text Language Processors as classes to clean up, tokenize and tag texts of various languages."""

from sign_language_translator.languages.text.english import English
from sign_language_translator.languages.text.text_language import TextLanguage
from sign_language_translator.text.tagger import Tags

__all__ = [
    "English",
    "TextLanguage",
    "Tags",
]
