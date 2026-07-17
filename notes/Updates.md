So this is the first stop of this project. Not the end but a halt, why? 

Well because after loss we are now implementing backpropagation. 

I can simply learn and substitute the values for backward function. 

But,

I am deriving each and every backward function on my own to learn the math behing every function. 

## CURRENT PROGRESS

Currently, 

I am deriving the backward function for the Linear Layer in LM Head. 

For that we need the Cross Entropy Backward, Why?

Because for the Linear Layer we will be needing, 

$\frac{dL}{dweights}$, $\frac{dL}{dbiases}$ to improve the loss from linear layer and $\frac{dL}{dX}$ to pass to previous layer.

**Using the chain rule** we will have, 

$\frac{dL}{dweights} = \frac{dL}{dLogits} * \frac{dlogits}{dWeights}$

The term $\frac{dL}{dLogits}$ is given by the _Cross Entropy Loss_.

Because what happends in Backpropagation is that, each layer tells the previous layer that how much the previous layer affected the result of the current layer. 

That's why each layer sends a component backward to help the other layer improve its parameters as per the affected loss in the layer that sent it backward. 

In a simple manner, 

    Cross Entropy Loss

        │
        ▼
    dL/dLogits

        │
        ▼
    Linear Layer
        │
        ▼ 
    -->dL/dW 
   
    -->dL/db 
   
    -->dL/dX
        │
        ▼
      dL/dX
        │
        ▼
    Transformer

and so on.

Thats why we first calcualte the $\frac{dL}{dLogits}$.

However I am making some mistake in the derivation, I dont know where. And since I dont want any help from AI. I will solve it myself but it is taking time. So no implementaions will be done in code as of today. 

Current Derivation, 

Since, In the loss calculation we are performing the functions in this manner, 

Logits --> Softmax --> Probabilities --> Loss

So, Loss is dependent on Softmax i.e, $\frac{dL}{dSoftmax}$, and Softmax is dependent on Logits i.e., $\frac{dSoftmax}{dLogits}$.

Using Chain Rule we have, 

$\frac{dL}{dSoftmax} * \frac{dSoftmax}{dLogits} = \frac{dL}{dLogits}$

Exactly what we need. 

- $\frac{dSoftmax}{dLogits}$ has two cases;

    - CASE I: $i = j$ then,

        $\frac{dSoftmax}{dLogits} = {S_{z_{i}}} * (1 - {S_{z_{i}}})$

    - CASE II: $i \ne j$ then, 

        $\frac{dSoftmax}{dLogits} = -{S_i} * {S_{j}}$

- $\frac{dL}{dSoftmax}$

    - $ Loss = - \sum_{i} y_i \ln(Softmax)$

        $\frac{dL}{dSoftmax} = - \frac{y_i}{Softmax}$


After using the chain rule I am getting 
 
$\frac{dL}{dLogits} = z_i * (Softmax_i - 1)$ For CASE I. 

$\frac{dL}{dLogits} = Softmax_j * z_i$ For CASE II.

I read the documentation of BERT and found that it will be equal to $Softmax_i - Target$. 

I need to somehow merge the two cases, but I am unable to do so, maybe it require something I dont know. I am confused. 

Got the answer, And I will be real, I did google to find what is a function that can combine two cases together and it first gave me 

**Indicator Function**

But that didnt work out too well.

So after a little bit more research on wiki, can across

**Kronecker delta**

Than it stuck me that i studies a little bit about kronecker delta when I was making my FNN last year. 

Now I have the fully dervied equation, 

$\frac{dL}{dZ_i} = S_i - y_i$


## OPTIMIZER

Run the optimizer once normally and gave,

8.20811257029584 - first pass

8.079217435084834 - after SGD second pass

Now before moving on to the Transformer Backward Pass, I would run epochs and test how far can we bring down the loss.

My institution is learning a single sentence "I LOVE DOGS" can be done by just updating LM Head weights and biases. Basically testing if we can over-fit it on a single sentence. 

So I will be changing the corpus to a single sentence. 

Let's see.


Amazed to see that after 100 epoches we are getting the loss = 1.1504247663774732. 

So I went back and ran a test for 74 seconds and the loss upon overfitting was 0.225074134944918, from LM Head updation alone. 

It's proof that the optimization pipeline works.

## BUG

A bug was found during the sanity check. 

The bug was I forget to relate the Vocabulary size to the tokenizer.

Hence the vocab size was 5000, and tokenizer has only 8-9 words in vocab, leading the model to prediciting _Ghost Words_ which never exist. 

Now it is fixed, the embedding layer and LM Head depend on tokenizer for vocab size. 

As a result the loss has decreased significantly. 

At this point, we've built a complete trainable neural network.

It's just that only the last layer is trainable. 😂

Okay, so we have added the Feed Forward backward pass as well. 

Now it's time for the layer norm. I thought it might be a walk in the park like the previous backward passes. 

But man I was wrong. 

Okay so in forward pass, we are doing something like this $self.gamma * normalized_X + self.beta$

You see here,

self.gamma is of shape (128,1)

and

normalized_X is of shape (10, 128)

Mathematically, it is not possible to multiply them. 

But NumPy uses broadcasting and duplicates the entire single column 10 times to match the row of normalized_X.

Known as element wise multipliaction. 

It was very confusing at first but now I understand it. 


Okay its getting quite serious now, 

I mean calculating, 

self.d_gamma = np.sum(doutput * self.normalized_X, axis=0)

self.d_beta = np.sum(doutput, axis=0)

d_normalized_X = doutput * self.gamma

This was easy but now since everything indirectly depends on X, and X influences both the mean, standard deviation and Normalized X. 

In order  to find the $\frac{dL}{dX}$ we will need to find the derivation of every path that depends on X. 


Backward
=
The input (X) first produces the mean and variance, which are then used to compute the standard deviation, and finally the normalized output. As a result, each input feature influences the loss in more than one way.

Hence we have two path to compute, 
$
\frac{\partial L}{\partial X}
$

**Path 1 (Direct)**

$\frac{\partial L}{\partial \text{Centered}}
\cdot
\frac{\partial \text{Centered}}{\partial X}
$
+
$\frac{\partial L}{\partial \text{Variance}}
\cdot
\frac{\partial \text{Variance}}{\partial X}
$
+
$\frac{\partial L}{\partial \text{Mean}}
\cdot
\frac{\partial \text{Mean}}{\partial X}
$ = 
$
\frac{\partial L}{\partial X}
$

**Path 2 (Indirect)**

$\frac{\partial L}{\partial \hat{X}}
\cdot
\frac{\partial \hat{X}}{\partial X}$
+
$\sum_{k = 1}^{N}
\frac{\partial L}{\partial \hat{X_k}}
\cdot
\frac{\partial \hat{X_k}}{\partial \mu}
\cdot
\frac{\partial \mu}{\partial X}$
+
$\sum_{k = 1}^{N}
\frac{\partial L}{\partial \hat{X_k}}
\cdot
\frac{\partial \hat{X_k}}{\partial \sigma}
\cdot
\frac{\partial \sigma}{\partial X}$=
$
\frac{\partial L}{\partial X}
$

**_Backward Pass: Two Paths to dInput_**

_Path 1: Direct_

Calculates and stores intermediate gradients for every operation (d_mean, d_var, d_std). It is intuitive and easy to debug, but creates many temporary arrays, making it slow and memory-heavy.

_Path 2: Indirect_

Applies the multivariable chain rule and simplifies the math. The intermediate gradients (d_mean, d_var) cancel out, collapsing into a single, elegant equation. This is the standard implementation because it requires fewer operations and is significantly faster.


Personally I prefer Path 2 not only because it;s fast and few operation rather how

 _Paul Dirac famously stated that "It is more important to have beauty in one's equations than to have them fit experiment"_


 Now, I need to derive each one of them, so I will update backward function tomorrow. 
 
The backward function in Layer Norm looks absolutely beautiful. 

Now the main part is to merge them into the Encoder. Feed Forward and Layer Norm both.

Encoder backward has been fully coded. 

However, it did take quite some understanding and time to solve the softmax derivation in Self Attention. 

But at last we are here. Backward function for the attention is fully functionally now and has been merged to Encoder.
