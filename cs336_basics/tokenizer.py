import pickle
from collections.abc import Iterable

import regex as re  # pyright: ignore[reportMissingModuleSource]

from . import common


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self._vocab: dict[int, bytes] = vocab
        self._merges: list[tuple[bytes, bytes]] = merges
        self._special_tokens: list[str] | None = (
            sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else None
        )
        self._rvocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        if self._special_tokens:
            for s in self._special_tokens:
                if s.encode("utf-8") not in self._vocab.values():
                    self._vocab.update({len(self._vocab), s.encode("utf-8")})
                    self._rvocab.update({s.encode("utf-8"), len(self._vocab)})

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        if self._special_tokens is None:
            paragraphs = [text]
        else:
            pattern = r"(" + "|".join(re.escape(t) for t in self._special_tokens) + r")"
            paragraphs = re.split(pattern, text)
        for p in paragraphs:
            if self._special_tokens is not None and p in self._special_tokens:
                if p.encode("utf-8") in self._rvocab:
                    tokens.append(self._rvocab[p.encode("utf-8")])
                else:
                    print(f"special token {p} do not exist")
            else:
                for match in re.finditer(common.PAT, p):
                    word = match.group()
                    tokens.extend(self._encode_word(word))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for s in iterable:
            tokens = self.encode(s)
            yield from tokens

    def decode(self, ids: list[int]) -> str:
        words = [self._vocab[id] for id in ids]
        return b"".join(words).decode(encoding="utf-8", errors="replace")

    def _encode_word(self, word: str) -> list[int]:
        subbytes = [bytes([b]) for b in word.encode("utf-8")]
        for m in self._merges:
            if len(subbytes) <= 1:
                break

            tmp_bytes = []
            i = 0
            while i < len(subbytes) - 1:
                if m == (subbytes[i], subbytes[i + 1]):
                    tmp_bytes.append(subbytes[i] + subbytes[i + 1])
                    i = i + 2
                else:
                    tmp_bytes.append(subbytes[i])
                    i = i + 1
            if i == len(subbytes) - 1:
                tmp_bytes.append(subbytes[-1])

            subbytes = tmp_bytes

        tokens = [self._rvocab[b] for b in subbytes]
        return tokens
