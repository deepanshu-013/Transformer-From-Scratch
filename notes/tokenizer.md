## Mistake

Initially I rebuilt the vocabulary every time encode() was called.

**Problem:**

The same word could receive different IDs.

**Solution:**

Separate fit() from encode().

fit() builds the vocabulary once.

encode() only performs lookups.


## Mistake

The fit() is only compatible with string.

**Problem:** 

If a corpus is given, the function fails. 

**Solution:** 

I went with corpus as input, converting string to list using isinstance() function. 
Also used the update() attribute and made little optimizations. 


## Next Step

Next I added the special tokens in encode(), currently doing it from one sentence only, since its easy to implement and test. 

However, a question arose in my mind that when do we use the UNK and MASK special token. 

To which I find the answer, 

UNK: When an unknown word is given in encode, it uses UNK to address it instead of throwing an error.

MASK: It is used to train model to find what word could be at the place of mask using the sentence as a context.  (Interesting Stuff) 


## MISTAKE

It was a stupid on, embarrassed to even talk about it.

**Problem:** 

id_to_word{} was throwing KeyError : 2. Since it doesn't have a special tokens added to it. 

**Solution:** 

Used the ShortHand Loop in the id2word{} to generate special tokens value-key pair from word2id{}. 😂

## MISTAKE

**Problem:**

Encode() was encoding vocabulary, not the sentence. 

**Solution:**

Provided an input parameter for single sentence for now. And encode the sentence. 