import re

EXTRA_SYMBOLS = {"ʕ": 7, "ħ": 8}
_PHONEME_FIXUPS = {"̪": "", "ˤ": "", "[": "", "]": "", "{": "", "}": ""}
_SYLLABLE_DOT = re.compile(r"(?<=\S)\.(?=\S)")
_LATIN_RUN = re.compile(r"[A-Za-z]+")
_CITATION = re.compile(r"\[[^\]]*\]")
_WHITESPACE = re.compile(r"\s+")
_TASHKEEL = re.compile("[ً-ْٰ]")
_ARABIC_LETTER = re.compile("[ء-ي]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([،؛؟.!])")


def normalize_text(text: str) -> str:
    text = _CITATION.sub(" ", text)
    text = _LATIN_RUN.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def clean_phonemes(phonemes: str) -> str:
    phonemes = _SYLLABLE_DOT.sub("", phonemes)
    for old, new in _PHONEME_FIXUPS.items():
        phonemes = phonemes.replace(old, new)
    return phonemes


def is_diacritized(text: str, minimum_ratio: float = 0.2) -> bool:
    letters = len(_ARABIC_LETTER.findall(text))
    return letters > 0 and len(_TASHKEEL.findall(text)) / letters >= minimum_ratio


class ArabicDiacritizer:
    def __init__(self):
        from camel_tools.disambig.mle import MLEDisambiguator

        self._model = MLEDisambiguator.pretrained()

    def __call__(self, text: str) -> str:
        text = normalize_text(text)
        if not text or is_diacritized(text):
            return text

        from camel_tools.tokenizers.word import simple_word_tokenize

        output = []
        for result in self._model.disambiguate(simple_word_tokenize(text)):
            if result.analyses:
                output.append(result.analyses[0].analysis.get("diac", result.word))
            else:
                output.append(result.word)
        return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", " ".join(output))
