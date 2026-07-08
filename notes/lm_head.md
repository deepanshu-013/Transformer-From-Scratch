Before building the Masked Language Modeling (MLM) pipeline. There is a issue in the current state, i.e., 

We have the matix of shape (10, 128) after the Encoder. 

But in the vocabulary we have vocab_ids so we cant not compare the 128 features with the 5000 vocabulary id. 

Why build the LM Head first?

Imagine we skipped it.

Transformer output

(10,128)

Ground truth

dogs

How do you compare them?

You can't.

The LM Head creates the bridge.


Its quite beautiful how LM Head is making the model understand that _"Which word embedding is most similar to my hidden representation?"_

So we have two cases in this matter:

**Case 1: Weight Tying**

we can use the embedding matrix we created at the early stage and transpose it to weight matrix. 

because we have noticed that, 

the weight matrix in LMHead is of shape (128, 5000), while the matrix in embedding is of shape (5000, 128).

that suggests that weight_matrix = embedding_matrix.Transpose

It has its own advantage as it require less computation and more memeory efficient. 

**Case 2: Separate Linear Layer** 

we create an entire new weight matrix without using the embedding matrix.

the model learns everything itself without even requiring its previous representations. 

(I must say, that is very beautiful.)