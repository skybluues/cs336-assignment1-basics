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
    import time

    t0 = time.perf_counter()
    pretoken_map = pretokenization.pretokenize(input_path, special_tokens)
    print(f"pretoken: {time.perf_counter() - t0}s")
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
    t0 = time.perf_counter()
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
        pair_map.pop(topest_pair[0])
        """
        update subtokens_map and pair_map
        """
        to_pop: list[tuple] = []
        to_append: list[tuple[tuple, int]] = []
        for subtokens, count in subtokens_map.items():
            if len(subtokens) < 2:
                to_pop.append(subtokens)
                continue
            to_merge: list[tuple[int, int]] = []
            subtokens_new = []
            i = 0
            while i < len(subtokens) - 1:
                if (subtokens[i], subtokens[i + 1]) == merge_list[-1]:
                    subtokens_new.append(b"".join(merge_list[-1]))
                    to_merge.append((i, i + 1))
                    i += 2
                else:
                    subtokens_new.append(subtokens[i])
                    i += 1
            if i == len(subtokens) - 1:
                subtokens_new.append(subtokens[-1])
            if len(to_merge) > 0:
                to_pop.append(subtokens)
                to_append.append((tuple(subtokens_new), count))
            for i in range(len(to_merge)):
                if to_merge[i][1] < len(subtokens) - 1:
                    if (
                        subtokens[to_merge[i][1]],
                        subtokens[to_merge[i][1] + 1],
                    ) != topest_pair[0]:
                        pair_map[(subtokens[to_merge[i][1]], subtokens[to_merge[i][1] + 1])] -= count
                        if (
                            pair_map[
                                (
                                    subtokens[to_merge[i][1]],
                                    subtokens[to_merge[i][1] + 1],
                                )
                            ]
                            <= 0
                        ):
                            pair_map.pop(
                                (
                                    subtokens[to_merge[i][1]],
                                    subtokens[to_merge[i][1] + 1],
                                )
                            )
                    if i < len(to_merge) - 1 and to_merge[i][1] + 1 == to_merge[i + 1][0]:
                        pair_map[(b"".join(topest_pair[0]), b"".join(topest_pair[0]))] = (
                            pair_map.get((b"".join(topest_pair[0]), b"".join(topest_pair[0])), 0) + count
                        )
                    else:
                        pair_map[(b"".join(topest_pair[0]), subtokens[to_merge[i][1] + 1])] = (
                            pair_map.get(
                                (
                                    b"".join(topest_pair[0]),
                                    subtokens[to_merge[i][1] + 1],
                                ),
                                0,
                            )
                            + count
                        )

                if to_merge[i][0] > 0:
                    if i > 0 and to_merge[i][0] - 1 == to_merge[i - 1][1]:
                        pass
                    else:
                        if (
                            subtokens[to_merge[i][0] - 1],
                            subtokens[to_merge[i][0]],
                        ) != topest_pair[0]:
                            pair_map[
                                (
                                    subtokens[to_merge[i][0] - 1],
                                    subtokens[to_merge[i][0]],
                                )
                            ] -= count
                            if (
                                pair_map[
                                    (
                                        subtokens[to_merge[i][0] - 1],
                                        subtokens[to_merge[i][0]],
                                    )
                                ]
                                <= 0
                            ):
                                pair_map.pop(
                                    (
                                        subtokens[to_merge[i][0] - 1],
                                        subtokens[to_merge[i][0]],
                                    )
                                )
                        pair_map[(subtokens[to_merge[i][0] - 1], b"".join(topest_pair[0]))] = (
                            pair_map.get(
                                (
                                    subtokens[to_merge[i][0] - 1],
                                    b"".join(topest_pair[0]),
                                ),
                                0,
                            )
                            + count
                        )

        for x in to_pop:
            subtokens_map.pop(x)
        for x in to_append:
            subtokens_map[x[0]] = subtokens_map.get(x[0], 0) + x[1]
    print(f"merge: {time.perf_counter() - t0}s")
    return vocab_map, merge_list


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "temp" / "one.txt"
    print(train_bpe(data_path, 500, ["<|endoftext|>"])[1])
