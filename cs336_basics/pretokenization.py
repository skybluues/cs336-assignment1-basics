import os
from typing import BinaryIO
import regex as re
from concurrent.futures import ProcessPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(input_path: str, bound: tuple[int, int], special_token: str) -> dict[str, int]:
    pretoken: dict[str, int] = {}
    with open(input_path, "rb") as file:
        file.seek(bound[0])
        chunk = file.read(bound[1] - bound[0]).decode("utf-8", errors="ignore")
        slice = chunk.split(special_token)
        for s in slice:
            for match in re.finditer(PAT, s):
                word = match.group()
                pretoken[word] = pretoken.get(word, 0) + 1
    return pretoken


def pretokenize(input_path: str, special_token: str) -> dict[bytes, int]:
    pretoken_all: dict[bytes, int] = {}
    num_processes = 8
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, special_token.encode())
    bound = list(zip(boundaries[:-1], boundaries[1:]))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(pretokenize_chunk, [input_path] * len(bound), bound, [special_token] * len(bound)))

    for res in results:
        for key, value in res.items():
            pretoken_all[key.encode("utf-8")] = pretoken_all.get(key.encode("utf-8"), 0) + value

    return pretoken_all


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[1] / "temp" / "one.txt"
    x = pretokenize(data_path, "<|endoftext|>")
    print(x)
