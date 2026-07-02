Since the model doesn't know which word is first.

Positional encoding fixes that.

Unlike NN, Transformer works in parallelism, so we have to let it know what comes first. 

For that purpose we will be using sinusoidal waves, since they retain what came first and next.

Plus it has **Uniqueness** and **Robustness** for long sentences, since the pattern is repeating and can be extracted from a simple function. 

## Problem

Currently, need some time to think clearly how to implement the encoding. 

I did found a solution, but it takes exponential time, which is just unacceptable for me. So I am still looking for a linear or at least an O(log n) or O(n log n), whatever just no exponential. 
