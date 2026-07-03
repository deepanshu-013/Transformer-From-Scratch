# a =[]
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


import numpy as np

a = np.array(([1, 2, 3, 4],
              [5, 6, 7, 8]))



print(a.shape)

for i in range(a.shape[0]):
    sin_count = 0
    cos_count = 0
    for j in range(a.shape[1]):
        if j % 2 == 0:
            sin_count += 1
            print(sin_count)
        else:
            cos_count += 1





























