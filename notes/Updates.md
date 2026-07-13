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