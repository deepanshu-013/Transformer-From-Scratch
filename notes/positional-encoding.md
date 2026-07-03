Since the model doesn't know which word is first.

Positional encoding fixes that.

Unlike NN, Transformer works in parallelism, so we have to let it know what comes first. 

For that purpose we will be using sinusoidal waves, since they retain what came first and next.

Plus it has **Uniqueness** and **Robustness** for long sentences, since the pattern is repeating and can be extracted from a simple function. 

## Problem

Currently, need some time to think clearly how to implement the encoding. 

I did found a solution, but it takes exponential time, which is just unacceptable for me. So I am still looking for a linear or at least an O(log n) or O(n log n), whatever just no exponential. 

## Solution 

We can vectorize it, which sovle the complexity problem. 

But I still went with the nested loops, for simplicity purpose, I will update it once the whole transformer is working. 

So current Time complexity is:

**_O(sequence length x embedding dimension)_**


## Discovery

One more thing I discovered that made me quite happy and little embarressed. 

In the formula for positional encoding using sinusoidal waves, the 'i' has a separate count for sin and cos, i.e.,

If we use sin for the 0th index we take i = 0, and for the 1st index we will use cos with i = 0. 

So 'i' increase the number of time we use the function, therefore i went with a 'Sin_count' & 'Cos_count' varaible first hand. 

But then I discovered something that, we don't need that, we can simply use the 'dim' in dimension, to find the count for sin and cos. 

For sin, i = dim/2, and since we need 2 * i in the formula, we can simply take 2i = 2 * dim/2 = dim. 

For cos, i = (dim - 1)/2, and for 2i = dim - 1. 

Which was a moment worth celebrating for me at least. 🥳