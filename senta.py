import spacy
import numpy as np

spacy.tokens.Token.set_extension("data", default=False, force=True)


class Senta:

    def __init__(self, nlp=None):
        if nlp is None:
            self.__nlp = spacy.load("en_core_web_lg")
        else:
            self.__nlp = nlp

        self.__lexicon = self.__load_list()

    def __load_list(self):
        result = {}
        with open("./sentiments/subjectivity.ttf") as f:
            for line in f.readlines():
                line_result = {}
                for entry in line.split(" "):
                    temp = entry.split("=")
                    if len(temp) == 2:
                        line_result[temp[0]] = temp[1].replace("\n", "")
                result[line_result['word1']] = line_result

        doc = self.__nlp(' '.join([x for x in result.keys() if '-' not in x]))
        for token in doc:
            token._.data = result[token.text]

        return doc

    def __review_token(self, token: spacy.tokens.Token) -> bool:
        if token.is_punct:
            return False

        if token.pos_ in ['NOUN', 'ADJ', 'VERB']:
            return True

        if token.dep_ == 'neg':
            return True

    def __review_wordlist(self, sentence: str) -> str:
        """Removes stopwords and punctuation characters from a string of words

        Loads a string into a spacy.tokens.doc.Doc and removes all stopwords
        and punctuation characters. The remaining words are lemmatized.

        For example: 'Everything was fresh and delicious!' turns into 'fresh
        delicious'.

        Args:
            sentence: The sentence that should be filtered

        Returns:
            The supplied str without any stopwords or punctuation characters.
        """
        doc = self.__nlp(sentence)
        filtered_words = [token.lemma_ for token in doc
                          if self.__review_token(token)]

        return ' '.join(filtered_words)

    def __is_pos_matching(self, token, word):
        token_pos = token._.data['pos1']
        if token_pos == 'anypos':
            return True

        return token_pos.lower() == word.pos_.lower()

    def __most_similar(self, word: spacy.tokens.token.Token):
        result = {}
        if word.vector_norm:
            queries = {
                'negative': [],
                'positive': []
            }
            for token in self.__lexicon:
                if token.prob > -15 and token.vector_norm and \
                        self.__is_pos_matching(token, word):
                    sentiment = token._.data['priorpolarity']
                    similarity = word.similarity(token)
                    if sentiment == 'both':
                        queries['negative'].append(similarity)
                        queries['positive'].append(similarity)
                    elif sentiment != 'neutral':
                        queries[sentiment].append(similarity)

            for sentiment, result_list in queries.items():
                result[sentiment] = sorted(result_list, key=lambda w: w,
                                           reverse=True)[:10]

        return result

    def analyze(self, word_list):
        result = {
            'negative': [],
            'positive': []
        }
        for word in self.__nlp(self.__review_wordlist(word_list)):
            similar_words = self.__most_similar(word)

            if len(similar_words) > 0:
                for sentiment, s_word in similar_words.items():
                    result[sentiment].extend(s_word)

        negative = np.mean(sorted(result['negative'], reverse=True)[:10])
        positive = np.mean(sorted(result['positive'], reverse=True)[:10])

        if negative > positive:
            return 0
        elif negative < positive:
            return 1
        else:
            return -999
