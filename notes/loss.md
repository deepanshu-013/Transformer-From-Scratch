Now the exciting part, 

**THE LOSS FUNCTION.** 

So in order for the values to make sense we will use loss fucntions to tell us how far off we are from the target. 

Using that loss we will update the weights as per the gradient.

Internally the loss function will look like this, 

Logits

↓

Softmax

↓

Probabilities

↓

Cross Entropy

↓

Loss

Now lets build it. 


## MISTAKE

Well there were several mistakes in implementing the softmax and Cross Entropy Loss. 

Main issue I was facing was I was considering the whole matrix instead of looking at the only raw that has the mask. 

Secondly the issue was with the softmax, I stabilized the numerator by subtracting the maximum, but didn't stabilize the denominator.

So the probabilities were incorrect. 

However fixing this lead to something mindblowing (to me atleast), computers are imperfect as well. 

When I calcualted the softmax.sum() after correct implementation I got 0.99999999999999 as the answer, which mathematically is 1.0. 

But due to float64 representation as per the IEEE 754 standard, computers represent the number as close as possible to the actual answer because many decimal numbers cannot be represented exactly in binary.

In playground.py I calculated 0.1 + 0.2 to personally test it and the answer I got was 0.30000000000000004. Closest binary representation. 

_Amazing!_


