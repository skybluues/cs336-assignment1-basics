import pickle
import resource
import time
from pathlib import Path

from . import train_bpe

if __name__ == "__main__":
    root_path = Path(__file__).resolve().parents[1]
    input_txt = root_path / "data" / "owt_train.txt"
    output_vocab = root_path / "output" / "owt.pkl"
    output_merges = root_path / "output" / "owt.pkl"

    t0 = time.perf_counter()
    vocab, merges = train_bpe.train_bpe(input_txt, 32000, ["<|endoftext|>"])
    duration = time.perf_counter() - t0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"duration: {duration:.3f}s")
    print(f"memory: {peak_kb}kb")

    maxtoken: bytes = b""
    for token in vocab.values():
        if len(token) > len(maxtoken):
            maxtoken = token

    print(f"maxlen_token: {maxtoken}, length: {len(maxtoken)}")

    with open(output_vocab, "wb") as f:
        pickle.dump(vocab, f)
    with open(output_merges, "wb") as f:
        pickle.dump(merges, f)
