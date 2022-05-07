import re


class _Tokenizer(object):
    """The abstract class of Tokenizer

    Implement ```tokenize``` method to split a string of sentence into tokens.
    Implement ```detokenize``` method to combine tokens into a whole sentence.
    ```special_tokens``` stores some helper tokens to describe and restore the tokenizing.
    """

    def __init__(self):
        pass

    def tokenize(self, sent):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError


class WordTokenizer(_Tokenizer):

    def __init__(self):
        super(WordTokenizer, self).__init__()

    def tokenize(self, sent):
        return sent.strip().split()

    def detokenize(self, tokens):
        return ' '.join(tokens)


class BPETokenizer(_Tokenizer):

    def __init__(self):
        """ Byte-Pair-Encoding (BPE) Tokenizer

        Args:
            codes: Path to bpe codes. Default to None, which means the text has already been segmented  into
                bpe tokens.
        """
        super(BPETokenizer, self).__init__()

    def tokenize(self, sent):
        return sent.strip().split()

    def detokenize(self, tokens):

        return re.sub(r"@@\s|@@$", "", " ".join(tokens))
        # return ' '.join(tokens).replace("@@ ", "")


class Tokenizer(object):

    def __new__(cls, type):
        if type == "word":
            return WordTokenizer()
        elif type == "bpe":
            return BPETokenizer()
        else:
            print("Unknown tokenizer type {0}".format(type))
            raise ValueError
