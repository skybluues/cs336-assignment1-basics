import heapq
import os
from pathlib import Path
import time

from . import pretokenization


class Rev:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __lt__(self, other):
        return self.x > other.x  # 原来更大的，在堆里反而更小


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode()
    merges: list[tuple[bytes, bytes]] = []

    pretoken = pretokenization.pretokenize(input_path, special_tokens)
    subword_ids: list[tuple[list[int], int]] = []  # [(subword1, count), (subword2, count)]
    pairs: dict[tuple[int, int], tuple[int, set[int]]] = {}  # (token1, token2) -> (count, [index_of_subword_ids])

    t0 = time.perf_counter()

    for word_bytes, count in pretoken.items():
        ids = [b for b in word_bytes]
        subword_ids.append((ids, count))
        for pair in zip(ids[:-1], ids[1:]):
            if pair in pairs:
                num, indexes = pairs[pair]
                indexes.add(len(subword_ids) - 1)
                pairs[pair] = (num + count, indexes)
            else:
                pairs[pair] = (count, {len(subword_ids) - 1})

    heap: list[tuple[int, Rev, tuple[int, int]]] = []
    for k, v in pairs.items():
        heap.append((-v[0], Rev((vocab[k[0]], vocab[k[1]])), (k[0], k[1])))
    heapq.heapify(heap)

    while len(vocab) < vocab_size and len(pairs) != 0 and len(heap) > 0:
        """ rebuild heap """
        if len(heap) > len(pairs) * 20:
            heap: list[tuple[int, Rev, tuple[int, int]]] = []
            for k, v in pairs.items():
                heap.append((-v[0], Rev((vocab[k[0]], vocab[k[1]])), (k[0], k[1])))
            heapq.heapify(heap)

        top = heapq.heappop(heap)
        finished = False
        while top[2] not in pairs or -top[0] != pairs[top[2]][0]:
            if len(heap) > 0:
                top = heapq.heappop(heap)
            else:
                finished = True
                break
        if finished:
            break

        pair = top[2]
        index = pairs[pair][1]

        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[len(vocab)] = b"".join((vocab[pair[0]], vocab[pair[1]]))
        pairs.pop(pair)

        for i in index:
            ids = subword_ids[i][0]
            total = subword_ids[i][1]
            new_ids = []
            m = 0
            while m < len(ids) - 1:
                if (ids[m], ids[m + 1]) == pair:
                    new_ids.append(len(vocab) - 1)

                    if len(new_ids) > 1:
                        key = (new_ids[-2], new_ids[-1])
                        count, indexes = pairs.get(key, (0, set()))
                        indexes.add(i)
                        pairs[key] = (count + total, indexes)
                        heapq.heappush(heap, (-(count + total), Rev((vocab[key[0]], vocab[key[1]])), key))
                        key = (ids[m - 1], ids[m])
                        count, indexes = pairs.get(key, (0, set()))
                        pairs[key] = (count - total, indexes)
                        heapq.heappush(heap, (-(count - total), Rev((vocab[key[0]], vocab[key[1]])), key))
                        if count - total <= 0:
                            pairs.pop(key)

                    m += 2
                else:
                    new_ids.append(ids[m])

                    if len(new_ids) > 1 and new_ids[-2] == len(vocab) - 1:
                        key = (new_ids[-2], new_ids[-1])
                        count, indexes = pairs.get(key, (0, set()))
                        indexes.add(i)
                        pairs[key] = (count + total, indexes)
                        heapq.heappush(heap, (-(count + total), Rev((vocab[key[0]], vocab[key[1]])), key))
                        key = (ids[m - 1], ids[m])
                        count, indexes = pairs.get(key, (0, set()))
                        pairs[key] = (count - total, indexes)
                        heapq.heappush(heap, (-(count - total), Rev((vocab[key[0]], vocab[key[1]])), key))
                        if count - total <= 0:
                            pairs.pop(key)

                    m += 1
            if m == len(ids) - 1:
                new_ids.append(ids[-1])

                if len(new_ids) > 1 and new_ids[-2] == len(vocab) - 1:
                    key = (new_ids[-2], new_ids[-1])
                    count, indexes = pairs.get(key, (0, set()))
                    indexes.add(i)
                    pairs[key] = (count + total, indexes)
                    heapq.heappush(heap, (-(count + total), Rev((vocab[key[0]], vocab[key[1]])), key))
                    key = (ids[m - 1], ids[m])
                    count, indexes = pairs.get(key, (0, set()))
                    pairs[key] = (count - total, indexes)
                    heapq.heappush(heap, (-(count - total), Rev((vocab[key[0]], vocab[key[1]])), key))
                    if count - total <= 0:
                        pairs.pop(key)

            subword_ids[i] = (new_ids, total)

        """ prober """
        if len(vocab) % 100 == 0:
            print(f"vocab_size[{len(vocab)}] merges_size[{len(merges)}]", flush=True)
            print(f"pairs_size[{len(pairs)}] heap_size[{len(heap)}]", flush=True)
            print(f"time_spent[{time.perf_counter() - t0:.3f}s]", flush=True)
            print("======", flush=True)

    duration = time.perf_counter() - t0
    print(f"train_bpe duration: {duration:.3f}s")

    return vocab, merges
