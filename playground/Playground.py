# a =[]
import numpy as np

b = {
    'cls': 0,
    'sep': 1,
    'unk': 2,
}
#
# ids = [0, 1, 3]
#
#
# for _ in range(len(ids)):
#     a.extend([k for k, v in b.items() if v == ids[_]])
#
# print(" ".join(a))


# text = "I love Dogs love."
# text = text.lower()
# text = text.replace(".", "")
# text = text.split()
# print(text)
#
# for word in text:
#     a.append(b[word])
#
# print(a)

# unique = sorted(set(text))
#
# for idx, word in enumerate(unique):
#     b[word] = idx + 3
#
# print(b)

corpus = [
    "I love dogs.",
    "Dogs bark.",
    "Cats meow."
]

# corpus = [ "I love dogs.", "Dogs bark.", "Cats meow." ] if corpus is list: cant access the if statement even though corpus is list
# instead we can use type(corpus),
# but best practice is: isinstance()

# unique = set()
#
# if isinstance(corpus, str):
#     corpus = [corpus]
#
# for i in corpus:
#     j = i.lower().replace('.', '').split()
#     unique.update(j)
#
# print(unique)



    # print('lol')
    # string = ''
    # for i in corpus:
    #     text = i.lower().replace('.', '').split()
    #     print(text)
    #     string += ''.join(text)
    #     print(string)


# for idx, word in enumerate(sorted(unique)):
#     c[word] = idx + 5


# c= [1, 2, 3]
#
# length = len(c)
#
# # print(" ".join(c))
#
# max_length = 8
# padding = max_length - length

# for i in range(padding):
#     c.append(0)
#
# print(c)
#
# c[0:length:1] = [1 for _ in range(length)]
#
# print(c)


#Need something clever...
# c.extend([0] * padding)
# print(c)
#
# attention = [1] * length + [0] * padding
# print(attention)

#Much Better, I guess.


# import numpy as np

# a = np.array([1.0, 0.36787944, 0.13533528])
# print(a / np.sum(np.exp(a), axis=0, keepdims=True))

# print((a / (1.0+(0.36787944)+(0.13533528))).sum())

# print(0.1 + 0.2) #crazy answer find


# # print(a.shape)

# # for i in range(a.shape[0]):
# #     sin_count = 0
# #     cos_count = 0
# #     for j in range(a.shape[1]):
# #         if j % 2 == 0:
# #             sin_count += 1
# #             print(sin_count)
# #         else:
# #             cos_count += 1

# print(a[0])

class Tokenizer:
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.CLS = 2
        self.SEP = 3
        self.MASK = 4

tokenizer = Tokenizer()

input_ids = np.random.randn(128, 128)
print(input_ids.zeros_())

# masked_input_ids = []
# labels = []
#
# valid_positions = []
# num_to_mask = max(1, round((len(input_ids) - 2) * 0.15))
#
# for position, token_id in enumerate(input_ids):
#             masked_input_ids.append(token_id)
#             labels.append(-100)
#             if token_id not in [tokenizer.CLS, tokenizer.SEP, tokenizer.PAD]:
#                 valid_positions.append(position) #Before we were pushing the token id instead of the index.
#
# print(valid_positions)
# print(masked_input_ids)
# print(labels)
#
# import random as rn
# print("num to mask:", num_to_mask)
# mask_indices = rn.sample(valid_positions, num_to_mask)
# print(mask_indices)
#
# for idx in mask_indices:
#     masked_input_ids[idx] = tokenizer.MASK
#     labels[idx] = input_ids[idx] #Instead of changing the index of mask_indices, we are using the mask_indices as the index and masking the wrong word.
#
#
# print(masked_input_ids)
# print(labels)

##--------Good now its fixed----##

















