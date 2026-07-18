So on the first search I went with LR = 0.01 and epoches in range of 1000. 

On Epoch 30, the result was, 

Epoch 30, Loss 1.2922722900122603e-09

--------------Gradients are------------------

0.5266973283841411

1478.865117772747

And after that everything exploded,

Since LR = 0.01, so the model learned the single sentence at 30. 

But the gradient kept on changing weights which than exploded and became so large that SoftMax can't handle it anymore. 

Or it could be because of the error after Epoch 30, 

RuntimeWarning: divide by zero encountered in log

total_loss += -np.log(correct_class_prob)

So we need to clip the probability so it never becomes zero again and let me decrease the learning rate as well, LR = 0.001.

There is problem now, after one to many runs, 

the loss if getting saturated and changing the dimensions and checking everything does nothing. Even changing the learning rate from 0.001 to 0.01 barely changes the loss. 

I dont know what i causing the saturation. 
