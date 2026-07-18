import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels
        total_loss = 0
        self.total_mask_count = 0
        e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.softmax = e_logits / np.sum(e_logits, axis=1, keepdims=True)

        for position, label in enumerate(labels):
            if label == -100:
                continue
            else:
                correct_class_prob = self.softmax[position, label]
                correct_class_prob = np.clip(correct_class_prob, 1e-12, 1)
                total_loss += -np.log(correct_class_prob)
                self.total_mask_count += 1
        if self.total_mask_count == 0:
            return 0.0
        return total_loss / self.total_mask_count

    def backward(self):
        d_logits = np.zeros_like(self.softmax)
        for position, label in enumerate(self.labels):
                    if label == -100:
                        continue

                    d_logits[position] = self.softmax[position]
                    d_logits[position, label] -= 1

        return d_logits/self.total_mask_count

    
