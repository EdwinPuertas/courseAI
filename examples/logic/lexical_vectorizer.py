import re
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from logic.text_processing import TextProcessing
from logic.lexical import lexical_es, lexical_en


class LexicalVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_processing=None):
        try:
            if text_processing is None:
                self.tp = TextProcessing(lang=lang)
            else:
                self.tp = text_processing
            self.lexical = lexical_es if lang == 'es' else lexical_en
        except Exception as e:
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            lexical = self.lexical
            tags = ('mention', 'url', 'hashtag', 'emoji', 'rt')
            features = np.zeros((len(X), 28), dtype=np.float)
            for row, doc in enumerate(X):
                doc = doc.lower()
                tokens_text = TextProcessing.tokenizer(doc)
                features[row, 0] = sum(1 for word in tokens_text if word == 'mention')
                features[row, 1] = sum(1 for word in tokens_text if word == 'url')
                features[row, 2] = sum(1 for word in tokens_text if word == 'hashtag')
                features[row, 3] = sum(1 for word in tokens_text if word == 'emoji')
                features[row, 4] = sum(1 for word in tokens_text if word == 'rt')

                label_word = sum([features[row, i] for i in range(0, 5)])
                features[row, 5] = float(len(tokens_text) - label_word)

                features[row, 6] = sum(1 for word in tokens_text if word in lexical['first_person_singular'])
                features[row, 7] = sum(1 for word in tokens_text if word in lexical['second_person_singular'])
                features[row, 8] = sum(1 for word in tokens_text if word in lexical['third_person_singular'])
                features[row, 9] = sum(1 for word in tokens_text if word in lexical['first_person_plurar'])
                features[row, 10] = sum(1 for word in tokens_text if word in lexical['second_person_plurar'])
                features[row, 11] = sum(1 for word in tokens_text if word in lexical['third_person_plurar'])

                features[row, 12] = np.nanmean([len(word) for word in tokens_text if word not in tags])
                features[row, 12] = features[row, 12] if not np.isnan(features[row, 12]) else 0.0
                features[row, 12] = round(features[row, 12], 4)

                features[row, 13] = kurtosis([len(word) for word in tokens_text if word not in tags])
                features[row, 13] = features[row, 13] if not np.isnan(features[row, 13]) else 0.0
                features[row, 13] = round(features[row, 13], 4)

                features[row, 14] = skew(np.array([len(word) for word in tokens_text if word not in tags]))
                features[row, 14] = features[row, 14] if not np.isnan(features[row, 14]) else 0.0
                features[row, 14] = round(features[row, 14] , 4)

                # adverbios
                features[row, 15] = sum(1 for word in tokens_text if word in lexical['adverb_neg'])
                features[row, 16] = sum(1 for word in tokens_text if word in lexical['adverb_time'])
                features[row, 17] = sum(1 for word in tokens_text if word in lexical['adverb_place'])
                features[row, 18] = sum(1 for word in tokens_text if word in lexical['adverb_mode'])
                features[row, 19] = sum(1 for word in tokens_text if word in lexical['adverb_cant'])
                features[row, 20] = sum([features[row, i] for i in range(15, 21)])
                features[row, 21] = sum(1 for word in tokens_text if word in lexical['adjetives_neg'])
                features[row, 22] = sum(1 for word in tokens_text if word in lexical['adjetives_pos'])
                features[row, 23] = sum(1 for word in tokens_text if word in lexical['who_general'])
                features[row, 24] = sum(1 for word in tokens_text if word in lexical['who_male'])
                features[row, 25] = sum(1 for word in tokens_text if word in lexical['who_female'])
                #features[row, 26] = sum(1 for word in tokens_text if word in lexical['hate'])
                features[row, 26] = self.lexical_diversity(doc)

                
 ######################## Here Goes New Metrics               
                
                
                features[row, :] /= len(tokens_text)
            return features
        except Exception as e:
            print('Error get_lexical_features: {0}'.format(e))

    @staticmethod
    def lexical_diversity(text):
        result = None
        try:
            text_out = re.sub(r"[\U00010000-\U0010ffff]", '', text)
            text_out = re.sub(
                r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
                r'|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                '', text_out)
            text_out = text_out.lower()
            result = round((len(set(text_out)) / len(text_out)), 4)
        except Exception as e:
            print('Error lexical_diversity: {0}'.format(e))
        return result