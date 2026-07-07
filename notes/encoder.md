I have created a separate Encoder Class in 'encoder.py'. Why?

Well, because;

It reduces the complexity in the 'transformer.py' and for the purpose that we need to add the residual in the output, in order to retain the information. 

Making a complete new class is not ideal, and adding it manually in the transformer steps is a good practice either. 

**_So the encoder flow is;_**

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


## ADDITION

Layer Normalization has beed added completely in the Encoder. 

We are using the Post-LayerNorm, later we will use the Pre-LayerNorm and compare their results. 

So in the LayerNorm we have three things I learned that are worth mentioning, they are;

--> Epsilon: We are using epsilon to be 1e-6 because sometimes the variance becomes zero and dividing by zero is bad. O_o

--> Gamma: Also known as the scale, used with the beta to make adjustments to the normalized data.

--> Beta: Also known as the shift, used to shift the data by addition. 

Basically after normalization we have mean of 0 and variance of 1, I know I know, its is known as standardization, but in early 2017, the name normalization stuck around. Why? 

Well I dont know the history. Check it your self. >.<

Anyways, Layer Normalization is compeleted. 


## MISTAKE

I added the positional encoding inside the encoder block which was wrong, because when encoder will be called again and again, we will be adding the positional encoding again and again.

So I migrated the positional encoding back to the transformer.py, later will add it to the transformer class. 

I also update the flow diagram above in this .md as per the mistake.

