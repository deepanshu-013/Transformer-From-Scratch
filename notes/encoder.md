I have created a separate Encoder Class in 'encoder.py'. Why?

Well, because;

It reduces the complexity in the 'transformer.py' and for the purpose that we need to add the residual in the output, in order to retain the information. 

Making a complete new class is not ideal, and adding it manually in the transformer steps is a good practice either. 

**_So the encoder flow is;_**

Input Embeddings

        │
        ▼

add Positional Encoding

        │
        ▼
X

        │
        ▼
Self Attention

        │
        ▼
attention_output

        │
        ▼
Residual
(X + attention_output)

        │
        ▼
LayerNorm

        │
        ▼
FeedForward

        │
        ▼
Residual
(X + feedforward_output)

        │
        ▼
LayerNorm

        │
        ▼
Output