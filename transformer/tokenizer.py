class Tokenizer:
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.CLS = 2
        self.SEP = 3
        self.MASK = 4

    def fit(self, corpus):
        self.word_to_id = {'[PAD]': 0,
                           '[UNK]': 1,
                           '[CLS]': 2,
                           '[SEP]': 3,
                           '[MASK]': 4
                           }
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.unique = set()

        if isinstance(corpus, str):
            corpus = [corpus]

        for sentence in corpus:
            self.unique.update(sentence.lower().replace('.', '').split())

        for idx, word in enumerate(sorted(self.unique)):
            self.word_to_id[word] = idx + 5
            self.id_to_word[self.word_to_id[word]] = word


    def encode(self, sentence1):
        ids = [self.CLS]

        for text in sentence1.lower().replace('.', '').split():
            ids.append(self.word_to_id[text])

        ids.append(self.SEP)

        return ids

    def decode(self, token_ids):
        words = []

        for _ in token_ids:
            words.append(self.id_to_word[_])

        return words

    def brew(self, sentence, max_length, decode=False):

        dictionary = {}

        ids = self.encode(sentence)

        real_length = len(ids)
        padding = max_length - real_length

        if real_length <= max_length:

            ids.extend([self.PAD] * padding)

            dictionary["input_ids"] = ids
            dictionary["attention_mask"] = [1] * real_length + [0] * padding

        else:
            raise ValueError ("Maximum length is less than Token Length.")

        if decode:
            dictionary["tokens"] = self.decode(ids)

        return dictionary

