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

*After optimization, the performance has been significantly improved. After optimization, training on TinyStories (vocab_size=10000, special token <|endoftext|>) takes about 47 seconds in total: ~39s for pretokenization and ~8s for BPE merges. Peak RSS of the main process is about 177 MB. The longest token is b' accomplishment' (length 15, including a leading space), which is reasonable because frequent whole words with leading spaces are merged into single tokens. Previously, before optimizing the merge loop, training took about 6 minutes and was dominated by merging.*

**Problem(train_bpe_expts_owt): BPE Training on OpenWebText**

(a) I trained byte-level BPE on OpenWebText with vocab_size=32000 and special token <|endoftext|>, and saved OWTVocab.pkl / OWTMerges.pkl. Training took about 52 minutes (pretok ~17 min, merges ~35 min) with peak main-process RSS ~14.6 GB. The longest token is 64 bytes of repeated \xc3\x83\xc3\x82 (mojibake-like), alongside long runs of - / —. This is reasonable for noisy web text: frequent repeated byte patterns are merged into long tokens.

(b) Compared to TinyStories (10k vocab, ~47s, ~177MB, longest token b' accomplishment'), the OWT tokenizer has a larger vocabulary, needs much more time/memory, and its longest tokens are repetitive encoding/punctuation noise rather than English words. TinyStories tokens look like clean lexical units; OWT tokens reflect messy web data. So the two tokenizers encode different domain statistics, not just different sizes.
