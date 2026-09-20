**Problem(unicode1): Understanding Unicode**

(a) chr(0) returns '\x00'

(b) its printed representation has none

(c) its a non-print char. it can be represented by '\x00' in indepent context. while in string, its invisable.

**Problem(unicode2): Unicode Encodings**

(a) for unicode standard, utf-8 is the most popular encoding for most of the text from internet, and its friendly for English char as its the same with ASCII encoding keeping one byte for a char.

(b) for example: "牛来". there are many chars that should be encoded with more than one byte, so this function is wrong.

(c) b"\xFFFF". this sequence is not matched any unicode encoding.

**Problem(train_bpe): BPE Tokenizer Training**

Pass all unit tests.

**Problem(train_bpe_tinystories): BPE Training on TinyStories**

(a) It takes almost 6 minutes and 62 MB memory. The longest token is "accomplishment". I think it is reasonable. In addition, the memory usage is only for main process, not include all sub-processes during training.

(b) Merging pair and updating subtokens take the most time.
