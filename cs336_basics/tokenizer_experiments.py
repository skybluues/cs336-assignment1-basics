from pathlib import Path
import numpy as np
from array import array
from . import tokenizer


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parents[1]
    ts_txt = root_path / "data" / "TinyStoriesV2-GPT4-train.txt"
    owt = root_path / "data" / "owt_train.txt"
    ts_txt_valid = root_path / "data" / "TinyStoriesV2-GPT4-valid.txt"
    owt_valid = root_path / "data" / "owt_valid.txt"
    ts_vocab = root_path / "output" / "TinyStoriesVocab.pkl"
    ts_merges = root_path / "output" / "TinyStoriesMerges.pkl"
    owt_vocab = root_path / "output" / "OWTVocab.pkl"
    owt_merges = root_path / "output" / "OWTMerges.pkl"

    ts_tokenizer = tokenizer.Tokenizer.from_files(ts_vocab, ts_merges, ["<|endoftext|>"])
    owt_tokenizer = tokenizer.Tokenizer.from_files(owt_vocab, owt_merges, ["<|endoftext|>"])
    """
    ts_docs = []
    buf = ""
    with open(ts_txt, encoding="utf-8") as f:
        while len(ts_docs) < 10:
            chunk = f.read(1_000_000)
            if chunk == "":
                break
            buf += chunk
            while "<|endoftext|>" in buf and len(ts_docs) < 10:
                doc, buf = buf.split("<|endoftext|>", 1)
                if doc != "":
                    ts_docs.append(doc)
    owt_docs = []
    buf = ""
    with open(owt, encoding="utf-8") as f:
        while len(owt_docs) < 10:
            chunk = f.read(1_000_000)
            if chunk == "":
                break
            buf += chunk
            while "<|endoftext|>" in buf and len(owt_docs) < 10:
                doc, buf = buf.split("<|endoftext|>", 1)
                if doc != "":
                    owt_docs.append(doc)

    allts = "<|endoftext|>".join(ts_docs)
    allowt = "<|endoftext|>".join(owt_docs)

    # a
    ts_compress_ratio = len(allts.encode("utf-8")) / len(ts_tokenizer.encode(allts))
    owt_compress_ratio = len(allowt.encode("utf-8")) / len(owt_tokenizer.encode(allowt))
    print(f"ts_docs_len[{len(ts_docs)}], owt_docs_len[{len(owt_docs)}]")
    print(f"tsbytes[{len(allts.encode('utf-8'))}], owtbytes[{len(allowt.encode('utf-8'))}]")
    print(f"ts_tokenizer_compress_ratio[{ts_compress_ratio}], owt_tokenizer_compress_ratio[{owt_compress_ratio}]")

    print("======")

    # b
    ts_on_owt_ratio = len(allowt.encode("utf-8")) / len(ts_tokenizer.encode(allowt))
    print(f"ts_on_owt_ratio[{ts_on_owt_ratio}]")

    print("======")

    # c
    alltext = allts + allowt
    bytes_num = len(alltext.encode("utf-8"))
    t0 = time.perf_counter()
    owt_tokenizer.encode(alltext)
    duration = time.perf_counter() - t0
    throughput = bytes_num / duration
    estimate = 825 * 1024 * 1024 * 1024 / throughput  # seconds
    print(f"throughput[{throughput}], estimate[{estimate:.3f}s]")
    """

    # d
    def encode_file(in_path: str, out_path: str, tokenizer: tokenizer.Tokenizer):
        with open(in_path, encoding="utf-8") as src:
            with open(out_path, mode="wb") as dest:
                special_token = "<|endoftext|>"
                ids = array("H")
                buf = ""
                while True:
                    chunk = src.read(1_000_000)
                    if chunk == "":
                        break
                    buf += chunk
                    while special_token in buf:
                        part, buf = buf.split(special_token, 1)
                        ids.extend(tokenizer.encode(part + special_token))
                        if len(ids) >= 1_000_000:
                            ids.tofile(dest)
                            del ids[:]
                if len(buf) > 0:
                    ids.extend(tokenizer.encode(buf))
                if len(ids) > 0:
                    ids.tofile(dest)
                    del ids[:]

    encode_file(ts_txt, root_path / "output" / "TinyStoriesV2-GPT4-train.ids", ts_tokenizer)
    encode_file(ts_txt_valid, root_path / "output" / "TinyStoriesV2-GPT4-valid.ids", ts_tokenizer)
    encode_file(owt, root_path / "output" / "owt_train.ids", owt_tokenizer)
    encode_file(owt_valid, root_path / "output" / "owt_valid.ids", owt_tokenizer)
