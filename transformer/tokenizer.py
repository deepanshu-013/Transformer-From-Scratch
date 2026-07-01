class Tokenizer:

    def fit(self, corpus):
        self.PAD = 0
        self.UNK = 1
        self.CLS = 2
        self.SEP = 3
        self.MASK = 4
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
        print(words)

        return " ".join(words)