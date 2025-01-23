import re
import nltk
import spacy
import unicodedata
import requests
from spacy_syllables import SpacySyllables
from bs4 import BeautifulSoup
from nltk import TweetTokenizer
from spacy.lang.es import Spanish
from spacy.lang.en import English
from nltk.util import ngrams


class TextProcessing(object):
    name = 'Text Processing'
    lang = 'en'

    def __init__(self, lang: str = 'es'):
        self.lang = lang

    @staticmethod
    def nlp(text: str) -> list:
        try:
            list_tagger = []
            tp_nlp = TextProcessing.load_spacy(TextProcessing.lang)
            doc = tp_nlp(text.lower())
            print('original_text: {0}'.format(text))
            for token in doc:
                item = {'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_,
                        'dep': token.dep_, 'shape': token.shape_, 'is_alpha': token.is_alpha,
                        'is_stop': token.is_stop, 'is_digit': token.is_digit, 'is_punct': token.is_punct,
                        'syllables': token._.syllables}
                list_tagger.append(item)
            return list_tagger
        except Exception as e:
            print('Error nlp: {0}'.format(e))

    @staticmethod
    def load_spacy(lang: str) -> object:
        try:
            component = spacy.load('es_core_news_sm') if lang == 'es' else spacy.load('en_core_web_sm')
            SpacySyllables(component)
            component.add_pipe('syllables', last=True)
            print('- Text Processing: {0}'.format(component.pipe_names))
            return component
        except Exception as e:
            print('Error load spacy: {0}'.format(e))

    @staticmethod
    def proper_encoding(text: str) -> str:
        try:
            text = unicodedata.normalize('NFD', text)
            text = text.encode('ascii', 'ignore')
            return text.decode("utf-8")
        except Exception as e:
            print('Error proper_encoding: {0}'.format(e))

    @staticmethod
    def stopwords(text: str) -> str:
        try:
            nlp = Spanish() if TextProcessing.lang == 'es' else English()
            doc = nlp(text)
            token_list = [token.text for token in doc]
            sentence = []
            for word in token_list:
                lexeme = nlp.vocab[word]
                if not lexeme.is_stop:
                    sentence.append(word)
            return ' '.join(sentence)
        except Exception as e:
            print('Error stopwords: {0}'.format(e))

    @staticmethod
    def remove_patterns(text: str) -> str:
        try:
            text = re.sub(r'\©|\×|\⇔|\_|\»|\«|\~|\#|\$|\€|\Â|\�|\¬', '', text)
            text = re.sub(r'\,|\;|\:|\!|\¡|\’|\‘|\”|\“|\"|\'|\`', '', text)
            text = re.sub(r'\}|\{|\[|\]|\(|\)|\<|\>|\?|\¿|\°|\|', '', text)
            text = re.sub(r'\/|\-|\+|\*|\=|\^|\%|\&|\$', '', text)
            text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)
            return text.lower()
        except Exception as e:
            print('Error remove_patterns: {0}'.format(e))

    @staticmethod
    def clean(text: str, stopwords: bool = False) -> str:
        try:
            text_out = TextProcessing.proper_encoding(text)
            text_out = text_out.lower()
            text_out = re.sub("[\U0001f000-\U000e007f]", 'EMOJI', text_out)
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                '[URL]', text_out)
            text_out = re.sub("@([A-Za-z0-9_]{1,40})", 'MENTION', text_out)
            text_out = re.sub("#([A-Za-z0-9_]{1,40})", 'HASTAG', text_out)
            text_out = TextProcessing.remove_patterns(text_out)
            text_out = TextProcessing.stopwords(text_out) if stopwords else text_out
            text_out = re.sub(r'\s+', ' ', text_out).strip()
            text_out = text_out.rstrip()
            return text_out if text_out != ' ' else None
        except Exception as e:
            print('Error transformer: {0}'.format(e))

    @staticmethod
    def tokenizer(text: str) -> list:
        try:
            text_tokenizer = TweetTokenizer()
            return text_tokenizer.tokenize(text)
        except Exception as e:
            print('Error make_ngrams: {0}'.format(e))

    @staticmethod
    def make_ngrams(text: str, num: int):
        try:
            n_grams = ngrams(nltk.word_tokenize(text), num)
            return [' '.join(grams) for grams in n_grams]
        except Exception as e:
            print('Error make_ngrams: {0}'.format(e))

if __name__ == '__main__':
    tp_es = TextProcessing(lang='es')
    result_es = tp_es.nlp(
        'Ahora a la gente todo le parece tóxico, más si dices lo que sientes o te molesta…y NO, tóxico es quedarse '
        'callado por miedo a arruinar algo. Hay que aprender a quererse primero.')
    for i in result_es:
        print(i)

    tp_en = TextProcessing(lang='en')
    result_en = tp_en.nlp("The data doesn’t lie: here's what one of our teams learned when they tried a 4-day workweek.")
    for i in result_en:
        print(i)
