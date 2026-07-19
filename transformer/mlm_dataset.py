import numpy as np
import random

class MLMDataset:
    def prepare(self, input_ids, tokenizer):
        self.tokenizer = tokenizer
        masked = input_ids.copy()

        labels = [-100] * len(input_ids)

        candidates = []

        for i in range(1, len(input_ids) - 1):

            if input_ids[i] in (
                    self.tokenizer.PAD,
                    self.tokenizer.CLS,
                    self.tokenizer.SEP,
            ):
                continue

            candidates.append(i)

        random.shuffle(candidates)

        num_to_mask = max(1, int(round(len(candidates) * 0.15)))

        for idx in candidates[:num_to_mask]:

            labels[idx] = input_ids[idx]

            r = random.random()

            # 80% -> MASK
            if r < 0.8:
                masked[idx] = self.tokenizer.MASK

            # 10% -> random token
            elif r < 0.9:
                masked[idx] = random.randint(
                    5,
                    len(self.tokenizer.word_to_id) - 1
                )

            # 10% -> unchanged
            else:
                pass

        return np.array(masked), np.array(labels) # Were getting returned as list instead of np array.
    
















