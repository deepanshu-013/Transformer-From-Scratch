class Tokenizer:

    def fit(self, text):
        self.word_to_id = {'<PAD>': 0,
                           '<UNK>': 1,
                           '<CLS>': 2,
                           '<SEP>': 3,
                           '<MASK>': 4
                           }
        self.id_to_word = {}
#######----------------Bug: fit() should accept a corpus, not only the string-----------------------########
        if type(text) is list:
            for i in text:
                self.text = i.lower().replace('.', '').split()
                self.string = ''
                self.string += self.text

            for idx, word in enumerate(self.unique):
                self.word_to_id[word] = idx + 5

            for word in self.text:
                self.id_to_word[self.word_to_id[word]] = word

        else :
            self.text = text.lower().replace('.', '').split()
            self.unique = sorted(set(self.text))

        for idx, word in enumerate(self.unique):
            self.word_to_id[word] = idx + 5

        for word in self.text:
            self.id_to_word[self.word_to_id[word]] = word

    def encode(self):
        ids = []

        for text in self.text:
            ids.append(self.word_to_id[text])

        return ids

    def decode(self, token_ids):
        words = []

        for _ in token_ids:
            words.append(self.id_to_word[_])

        return " ".join(words)