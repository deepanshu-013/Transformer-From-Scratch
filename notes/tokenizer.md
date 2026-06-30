## Mistake

Initially I rebuilt the vocabulary every time encode() was called.

Problem:

The same word could receive different IDs.

Solution:

Separate fit() from encode().

fit() builds the vocabulary once.

encode() only performs lookups.


## Mistake

The fit() is only compatiable with string.

Problem: 

If a corpus is given, the function fails. 

Solution: 

