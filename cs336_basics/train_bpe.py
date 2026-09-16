import os
from pathlib import Path
from . import pretokenization


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for n in range(256):
        vocab[n] = bytes([n])
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode()
    return vocab


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_map = init_vocab(special_tokens)
    merge_list: list[tuple[bytes, bytes]] = []

    """
    1. create pre_tokens mapping count
    2. create subtoken sequence to count
    3. calculate the topest count of successive subtokens
    4. update subtoken sequence counter
    5. update topest counter
    6. repeat 3 until reach max size
    """

    pretoken_map = pretokenization.pretokenize(input_path, special_tokens[0])
    subtokens_map: dict[tuple, int] = {}
    for pretoken, count in pretoken_map.items():
        subtokens = tuple(bytes([b]) for b in pretoken)
        subtokens_map[subtokens] = count

    pair_map: dict[tuple, int] = {}
    for subtokens, count in subtokens_map.items():
        if len(subtokens) < 2:
            continue
        for pair in zip(subtokens[:-1], subtokens[1:]):
            pair_map[pair] = pair_map.get(pair, 0) + count

    while len(vocab_map) < vocab_size and len(pair_map) != 0:
        """
        find topest pair
        """
        topest_pair = max(pair_map.items(), key=lambda kv: (kv[1], kv[0]))
        """
        update vocab_map and merge_list
        """
        merge_list.append(topest_pair[0])
        vocab_map[len(vocab_map)] = b"".join(topest_pair[0])
        """
        update subtokens_map and pair_map
        """
        subtokens_map_new: dict[tuple, int] = {}
        pair_map_new: dict[tuple, int] = {}
        for subtokens, count in subtokens_map.items():
            if len(subtokens) < 2:
                continue
            subtokens_new = []
            i = 0
            while i < len(subtokens) - 1:
                if (subtokens[i], subtokens[i + 1]) == merge_list[-1]:
                    subtokens_new.append(b"".join(merge_list[-1]))
                    i += 2
                else:
                    subtokens_new.append(subtokens[i])
                    i += 1
            if i == len(subtokens) - 1:
                subtokens_new.append(subtokens[-1])
            subtokens_map_new[tuple(subtokens_new)] = count
            for pair in zip(subtokens_new[:-1], subtokens_new[1:]):
                pair_map_new[pair] = pair_map_new.get(pair, 0) + count

        subtokens_map = subtokens_map_new
        pair_map = pair_map_new

    return vocab_map, merge_list


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "temp" / "one.txt"
    print(train_bpe(data_path, 500, ["<|endoftext|>"])[1])
