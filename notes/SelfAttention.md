Now we are moving on to the most important part of the transformer. 

**The Attention mechanism.** 

For that, in the original Bert Model we have 768 / 64 = 12 attention head. 

Which is quite a lot, honestly. 

But for our transformer, we will use a **_single head_**, not _**multi-head.**_

For that, we will have only one head as per the name, and it will have a shape of (128, 128).
So GPU can compute it in one go. 

Later I will divide it into multi-head, I am currently thinking of 4 heads so we will have a shape of (128, 32).

For now lets build the single head. 

Nice and easy, no bugs we found, however there was little mistake I did, when I value the Q, K, V using gaussian distribution I forget to change the standard deviation from 1 to 0.02. 

But fixed it after a good sanity check. :)

Now I will be using a FNN which I made last year but haven't uploaded to GitHub yet. It's a **Feed Forward Neural Network** made by learning from **Sentdex**.

**_Thank you Sentdex. <3_**