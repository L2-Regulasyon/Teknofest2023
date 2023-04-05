import string
import unicodedata

from zemberek import (TurkishMorphology, TurkishSentenceExtractor,
                      TurkishSentenceNormalizer)

remove_punc = str.maketrans('', '', string.punctuation)


SPIECE_UNDERLINE = u"▁".encode("utf-8")


class TextNormalization:
    """Text normalization task
    """

    def __init__(self):
        """Constructor method
        """
        self.zemberek_morpholgy = TurkishMorphology.create_with_defaults()
        self.zemberek_normalizer = TurkishSentenceNormalizer(self.zemberek_morpholgy)
        self.zemberek_extractor = TurkishSentenceExtractor()

    def normalize(self,
                  text: str,
                  remove_space: bool = True,
                  do_lower_case: bool = True,
                  normalize_function: str = "NFKC",
                  is_turkish: bool = True,
                  use_zemberek: bool = True):
        """
        Preprocess text by removing extra space and normalizing via python-unicodedata library.
        
        ---------
        :param str text: Text for normalization
        :param bool remove_space: Whether remove empty spaces or not (defaults to True)
        :param bool do_lower_case: Whether do lower case or not (defaults to True)
        :param str normalize_function: Unicodedata normalize function. Either "NFC", "NFKC", "NFD" or "NFKD". (defaults to "NFKC")
        :param bool is_turkish: Whether text is in Turkish or not (defaults to True)
        :param bool use_zemberek: Whether to use Zemberek-Python's normalizer. Always do lowercase inside (defaults to True)
        :return: Normalized text
        """
        outputs: str = text

        if remove_space:
            outputs = " ".join(outputs.strip().split())

        outputs = unicodedata.normalize(normalize_function, outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])

        if use_zemberek:
            sentences = self.zemberek_extractor.from_paragraph(outputs)
            normalized_sentences = []
            for sentence in sentences:
                normalized_sentences.append(self.zemberek_normalizer.normalize(sentence))
            outputs = "".join(normalized_sentences)

        if do_lower_case:
            if is_turkish:
                outputs = outputs.replace('\u0049', '\u0131')  # I -> ı
                outputs = outputs.replace('\u0130', '\u0069')  # İ -> i

            outputs = outputs.casefold()

        return outputs


# Lowercasing with special token
def special_lowercase(x):
    """
    Add special token if text contains upper chars.
    
    ---------
    param x: Text
    return: Adjusted text
    """
    chars = []
    for char in x:
        if char.lower() != char:
            chars.append("# ")
            chars.append(char.lower())
        else:
            chars.append(char.lower())
    return "".join(chars)


# Word Count Feature
def feature_wordcount(x):
    """
    Count the word in a text using string split() function. If the length condition met, add special token
    
    ---------
    param x: Text
    return: Adjusted text
    """
    length = len(x.split())
    if length < 5:
        return "+ " + x
    elif 5 <= length < 10:
        return "++ " + x
    else:
        return "+++ " + x


def preprocess_text(textcol,
                    prevent_bias=0):
    """
    Remove punctuations, prevent the bias by bias level
    
    ---------
    param textcol: Text
    param prevent_bias: bias level. 2 means fully-unbiased, 1 means casing-unbiased, 0 means none bias prevention mechanism is being executed
    return: Adjusted text
    """
    # textcol.values[:] = [" ".join(elm.strip().split()) for elm in tqdm(textcol.values)]
    # textcol.values[:] = [unicodedata.normalize("NFKC", elm) for elm in tqdm(textcol.values)]
    # textcol.values[:] = ["".join([c for c in elm if not unicodedata.combining(c)]) for elm in tqdm(textcol.values)]

    # Removing punctuations
    textcol = textcol.apply(lambda x: x.translate(remove_punc))

    # Casing-Unbiased and Fully-Unbiased Flow
    if prevent_bias > 0:
        textcol = textcol.str.lower()

    # Vanilla Flow
    elif prevent_bias == 0:
        textcol = textcol.apply(special_lowercase)

    # textcol = textcol.apply(feature_wordcount)

    return textcol






