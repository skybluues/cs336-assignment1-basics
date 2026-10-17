import pickle
from collections.abc import Iterable
from collections.abc import Iterator

import regex as re  # pyright: ignore[reportMissingModuleSource]

from . import common


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self._vocab: dict[int, bytes] = dict(vocab)
        self._special_tokens: list[str] | None = (
            sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else None
        )
        self._rvocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        if self._special_tokens:
            for s in self._special_tokens:
                if s.encode("utf-8") not in self._rvocab:
                    maxid = max(self._vocab.keys())
                    self._vocab.update({maxid + 1: s.encode("utf-8")})
                    self._rvocab.update({s.encode("utf-8"): maxid + 1})
        self._merges_index: dict[tuple[bytes, bytes], int] = {}
        for i, merge in enumerate(merges):
            self._merges_index.update({merge: i})

        self._encoding_cache: dict[str, list[int]] = {}

    @classmethod
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
                    assert False
            else:
                for match in re.finditer(common.PAT, p):
                    word = match.group()
                    tokens.extend(self._encode_word(word))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            tokens = self.encode(s)
            yield from tokens

    def decode(self, ids: list[int]) -> str:
        words = [self._vocab[id] for id in ids]
        return b"".join(words).decode(encoding="utf-8", errors="replace")

    def _encode_word(self, word: str) -> list[int]:
        if word in self._encoding_cache:
            return self._encoding_cache.get(word)

        subbytes = [bytes([b]) for b in word.encode("utf-8")]
        while len(subbytes) > 1:
            pairs = zip(subbytes[:-1], subbytes[1:])
            board = [self._merges_index.get(x, 2**63 - 1) for x in pairs]
            if all(x == 2**63 - 1 for x in board):
                break
            target = min(board)
            index_set = []
            i = 0
            while i < len(board):
                if board[i] == target:
                    index_set.append(i)
                    i = i + 2
                else:
                    i = i + 1
            tmp_subbytes = []
            last_index = 0
            for index in index_set:
                if last_index != index:
                    tmp_subbytes.extend(subbytes[last_index:index])
                tmp_subbytes.append(subbytes[index] + subbytes[index + 1])
                last_index = index + 2
            if last_index < len(subbytes):
                tmp_subbytes.extend(subbytes[last_index:])
            subbytes = tmp_subbytes
        tokens = [self._rvocab[b] for b in subbytes]

        self._encoding_cache.update({word: tokens})
        return tokens
