#!/usr/bin/env python3
"""Sequence packing for materialized SFT JSONL.

Minimized rewrite aligned with internal sequence_packing_v2:
- supports multi-file input + streaming shuffle
- packs conversations by token budget
- uses unfinished-pack best-fit filling
"""

import argparse
import bisect
import glob
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

try:
    import orjson

    def _loads(x: str):
        return orjson.loads(x)

    def _dumps(x: Any) -> str:
        return orjson.dumps(x).decode("utf-8")
except Exception:  # pragma: no cover
    _loads = json.loads
    _dumps = json.dumps


def stream_samples(paths: List[str], buffer_size: int, seed: int, max_open_files: int) -> Iterable[Dict[str, Any]]:
    random.seed(seed)
    pending = paths[:]
    random.shuffle(pending)
    active = []

    def refill():
        while len(active) < max_open_files and pending:
            p = pending.pop(0)
            try:
                active.append(open(p, "r", encoding="utf-8"))
            except Exception:
                pass

    refill()
    buf: List[Dict[str, Any]] = []
    while active:
        to_close = []
        for f in active:
            try:
                line = next(f)
                item = _loads(line)
                buf.append(item)
                if len(buf) >= buffer_size:
                    random.shuffle(buf)
                    half = len(buf) // 2
                    for x in buf[:half]:
                        yield x
                    buf = buf[half:]
            except StopIteration:
                to_close.append(f)
            except Exception:
                continue

        for f in to_close:
            active.remove(f)
            f.close()
        refill()

    if buf:
        random.shuffle(buf)
        for x in buf:
            yield x


def _conv_key(messages: List[Dict[str, Any]]) -> str:
    return "".join(m.get("content", "") for m in messages if m.get("role") == "user")


def _token_count(item: Dict[str, Any]) -> int:
    if "num_generated_tokens" in item:
        return int(item["num_generated_tokens"])
    if "question_num_generated_tokens" in item and "answer_num_generated_tokens" in item:
        return int(item["question_num_generated_tokens"]) + int(item["answer_num_generated_tokens"])
    return -1


def pack_epoch(
    input_paths: List[str],
    output_path: str,
    maximum_token: int,
    tolerance: int,
    seed: int,
    buffer_size: int,
    full_pack_buffer: int,
    max_open_files: int,
) -> None:
    unfinished = []
    full_packs: List[List[Dict[str, Any]]] = []
    cur_pack: List[Dict[str, Any]] = []
    cur_tokens = 0
    cur_qset = set()
    cur_conv = 0

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_tokens = 0
    skipped_too_long = 0

    with out.open("w", encoding="utf-8") as fout:
        data_stream = stream_samples(
            input_paths, buffer_size=buffer_size, seed=seed, max_open_files=max_open_files
        )
        for item in tqdm(data_stream, desc=f"packing seed={seed}"):
            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                continue

            n_tok = _token_count(item)
            if n_tok < 0:
                skipped_no_tokens += 1
                continue
            if n_tok > maximum_token:
                skipped_too_long += 1
                continue

            q = _conv_key(messages)
            inserted = False
            for i, up in enumerate(unfinished):
                if up["left"] >= n_tok and q not in up["qset"]:
                    up["pack"].extend(messages)
                    up["tokens"] += n_tok
                    up["left"] -= n_tok
                    up["conv"] += 1
                    up["qset"].add(q)
                    picked = unfinished.pop(i)
                    if picked["left"] < tolerance:
                        full_packs.append(picked["pack"])
                    else:
                        ranks = [x["left"] for x in unfinished]
                        unfinished.insert(bisect.bisect(ranks, picked["left"]), picked)
                    inserted = True
                    break
            if inserted:
                continue

            if cur_tokens + n_tok > maximum_token and cur_pack:
                left = maximum_token - cur_tokens
                data = {
                    "pack": cur_pack,
                    "tokens": cur_tokens,
                    "left": left,
                    "conv": cur_conv,
                    "qset": cur_qset,
                }
                if left < tolerance:
                    full_packs.append(cur_pack)
                else:
                    ranks = [x["left"] for x in unfinished]
                    unfinished.insert(bisect.bisect(ranks, left), data)
                cur_pack = []
                cur_tokens = 0
                cur_qset = set()
                cur_conv = 0

            if q in cur_qset:
                continue
            cur_pack.extend(messages)
            cur_tokens += n_tok
            cur_qset.add(q)
            cur_conv += 1

            if len(full_packs) >= full_pack_buffer:
                random.shuffle(full_packs)
                half = len(full_packs) // 2
                for p in full_packs[:half]:
                    fout.write(_dumps({"messages": p}) + "\n")
                written += half
                full_packs = full_packs[half:]

        if cur_pack:
            full_packs.append(cur_pack)
        for up in unfinished:
            full_packs.append(up["pack"])

        random.shuffle(full_packs)
        for p in full_packs:
            fout.write(_dumps({"messages": p}) + "\n")
            written += 1

    print(
        f"pack done: output={output_path} written={written} "
        f"skip_no_tokens={skipped_no_tokens} skip_too_long={skipped_too_long}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Input jsonl files or globs.")
    parser.add_argument("--output", required=True, help="Output jsonl prefix/path.")
    parser.add_argument("--maximum-token", type=int, default=262144)
    parser.add_argument("--tolerance", type=int, default=1000)
    parser.add_argument("--starting-epoch", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--full-pack-buffer", type=int, default=100000)
    parser.add_argument("--max-open-files", type=int, default=110)
    args = parser.parse_args()

    input_paths: List[str] = []
    for pat in args.input:
        hits = sorted(glob.glob(pat))
        if hits:
            input_paths.extend(hits)
        else:
            input_paths.append(pat)
    input_paths = [p for p in input_paths if Path(p).is_file()]
    if not input_paths:
        raise FileNotFoundError("No valid input files found")

    for epoch in range(args.starting_epoch, args.starting_epoch + args.num_epochs):
        if args.num_epochs == 1:
            out = args.output
        else:
            out = args.output.replace(".jsonl", f".epoch_{epoch}.jsonl")
        pack_epoch(
            input_paths=input_paths,
            output_path=out,
            maximum_token=args.maximum_token,
            tolerance=args.tolerance,
            seed=epoch,
            buffer_size=args.buffer_size,
            full_pack_buffer=args.full_pack_buffer,
            max_open_files=args.max_open_files,
        )


if __name__ == "__main__":
    main()
