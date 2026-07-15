from tokenizer import Tokenizer
from embedding import Embedding
from encoder import Encoder
from positional_encoding import PositionalEncoding
from lm_head import LMHead
from mlm_dataset import MLMDataset
from loss import CrossEntropyLoss

## For testing the pipeline and chekcing is model is learning or not, later will add it to separate file. ##
class SGD:
    def __init__(self, learning_rate=0.001):
        self.learning = learning_rate
    def update(self, layer):
        layer.weights += -self.learning * layer.d_weights
        layer.biases += -self.learning * layer.d_biases

class Transformer:
    def __init__(self, num_layers = 1):
        self.encoders =  Encoder() 

    def forward(self, X):
        # for encoder in self.encoders:
        #     X = encoder.forward(X)
        # return X
        return self.encoders.forward(X)

# corpus = [
#     "I love dogs.",
#     "Dogs bark.",
#     "Cats meow."
# ]

token = Tokenizer()

mlm_dataset = MLMDataset()


position = PositionalEncoding()

transformer = Transformer(num_layers=2)

ce_loss = CrossEntropyLoss()

optimizer = SGD()

epoches = 0

token.fit('I LOVE DOGS') # Trained on single sentence to check the LM Head and the whole pipeline

embedding = Embedding(len(token.word_to_id), 128) # Made the embedding layer depend on the tokenizer

dic = token.brew('I LOVE DOGS', 10, decode=True)
input_ids = dic["input_ids"]
masked_input_ids, labels = mlm_dataset.prepare(input_ids) # labels will be used for loss calculation during training
input_weights = embedding.forward(masked_input_ids)
positional_encoding = position.forward(input_weights.shape[0], input_weights.shape[1])
X = positional_encoding + input_weights
X = transformer.forward(X)

lm_head = LMHead(len(token.word_to_id), embedding_dim=128) # Made the LM Head layer depend on the tokenizer

logits = lm_head.forward(X) 
loss, row = ce_loss.forward(logits, labels)

##--------------------------------------------- For debugging and checking the pipeline -----------------------------------------##
import numpy as np
# top_5 = np.sort(row)[-5:]
# print(top_5)
# top5_indices = np.argpartition(row, -5)[-5:]
# top5_list = top5_indices.tolist()
# print(top5_indices)
# for i in token.id_to_word:
#     if i in top5_list:
#         print(token.id_to_word[i])
# print(loss)

# while epoches != 100:
#     dlogits = ce_loss.backward()
#     lm_head.linear.backward(dlogits)
#     optimizer.update(lm_head.linear)
#     logits = lm_head.forward(X)
#     loss, row = ce_loss.forward(logits, labels)
#     epoches += 1
#     print(loss)

# top_5 = np.sort(row)[-5:]
# print(top_5)
# top5_indices = np.argpartition(row, -5)[-5:] # To get the top 5 values in descending order.
# top5_list = top5_indices.tolist()
# print(top5_indices)
# for i in token.id_to_word:
#     if i in top5_list:
#         print(token.id_to_word[i]) # Seeing what are the top 5 predicitions.

dlogits = ce_loss.backward()
d_X = lm_head.linear.backward(dlogits)
doutput = transformer.encoders.feedforward.backward(d_X)
print(doutput.shape)

