import random as rn
from tokenizer import Tokenizer


class MLMDataset:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def prepare(self, input_ids, mask_probability = 0.15):
        masked_input_ids = []
        labels = []
        num_to_mask = max(1, round((len(input_ids) - 2) * mask_probability))
        valid_positions = []

        for position, token_id in enumerate(input_ids):
            masked_input_ids.append(token_id)
            labels.append(-100)
            if token_id not in [self.tokenizer.CLS, self.tokenizer.SEP, self.tokenizer.PAD]:
                valid_positions.append(position)


        mask_indices = rn.sample(valid_positions, num_to_mask)
        for idx in mask_indices:
            masked_input_ids[idx] = self.tokenizer.MASK
            labels[idx] = input_ids[idx]

        return masked_input_ids, labels

    
















