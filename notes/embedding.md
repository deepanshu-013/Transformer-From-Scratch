Used np.seed() to generate the same distribution. 

As for the distribution, I used the Standard Gaussian Distribution with mean 0 and variance of 1.

As for the vocabulary size and the dimensions, I kept it as user-defined i.e. me, so i can later test how far can i push it on my PC. 

I wanted to Update the updates.md regarding the backward function of embeddings. 

So the Function uses, 

np.zeros for the d_weights, because in simple sense if we have a vocab of 5000, and only need 5 token. Updating other rows is useless since they never contribute anything to the final loss. 

So instead of using real number, we go with zeros since there gradient is zero.

For the update of the weights with respect to the tokens used, we are using add.at, since classical NumPy methods like array[indices] = output,

Overwrites a word gradient if it has appeared more than once. 

Hence, the previous gradient values for the token are lost. 

To fix this issue we are using add.at(). 