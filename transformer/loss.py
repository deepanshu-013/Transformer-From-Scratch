import numpy as np

class CrossEntropyLoss:
    @staticmethod
    def forward(logits, labels = [0]):
        loss = 0
        total_loss = 0
        total_mask_count = 0
        for position, label in enumerate(labels):
            if label == -100:
                continue
            else:
                e_logits = np.exp(logits[position] - np.max(logits[position], axis=0, keepdims=True))
                softmax = e_logits / np.sum(e_logits, axis=0, keepdims=True)
                print(softmax.sum())
                correct_class_prob = softmax[label]
                total_loss += -np.log(correct_class_prob)
                total_mask_count += 1
        if total_mask_count > 0:
            loss = total_loss / total_mask_count
        return loss

    