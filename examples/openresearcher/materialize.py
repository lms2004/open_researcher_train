#!/usr/bin/env python3
"""Materialize raw chat JSONL into chunked Megatron SFT records.

This is a minimized rewrite aligned with the internal materialize scripts:
- handles tool call args json normalization
- splits rendered chat template into role chunks
- supports thinking conversations by slicing from each user turn
"""

import argparse
import json
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm

try:
    import orjson

    def _loads(line: str):
        return orjson.loads(line)

    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def _loads(line: str):
        return json.loads(line)

    def _dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


_WORKER_TOKENIZER = None
_WORKER_INCLUDE_EXTRA = False
_WORKER_START_FROM_LAST_USER = True


def _init_worker(
    tokenizer_model: str,
    chat_template_text: Optional[str],
    trust_remote_code: bool,
    include_extra: bool,
    start_from_last_user: bool,
):
    global _WORKER_TOKENIZER
    global _WORKER_INCLUDE_EXTRA
    global _WORKER_START_FROM_LAST_USER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    _WORKER_INCLUDE_EXTRA = include_extra
    _WORKER_START_FROM_LAST_USER = start_from_last_user
    if chat_template_text:
        _WORKER_TOKENIZER.chat_template = chat_template_text


def replace_json_args(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool-call function arguments from json string to dict if needed."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []) or []:
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass
    return messages


def find_last_user_message_end(
    messages: List[Dict[str, Any]],
    tokenizer,
    enable_thinking: bool,
    tools: Optional[Any],
) -> int:
    """Find rendered character boundary right after the last user turn."""
    last_user_idx = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    if enable_thinking and (
        last_user_idx + 1 < len(messages)
        and messages[last_user_idx + 1].get("role") == "assistant"
        and not messages[last_user_idx + 1].get("reasoning_content")
    ):
        text = tokenizer.apply_chat_template(
            messages[: last_user_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        text += "<|im_start|>assistant\n<think></think>"
        return len(text)

    text = tokenizer.apply_chat_template(
        messages[: last_user_idx + 1],
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    return len(text)


def split_template_into_messages(
    messages: List[Dict[str, Any]],
    tokenizer,
    start_from_last_user: bool,
    enable_thinking: bool,
    tools: Optional[Any],
) -> List[Dict[str, str]]:
    """Split rendered template into per-turn chunks."""
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    out: List[Dict[str, str]] = []
    if start_from_last_user:
        system_end = full.find("<|im_end|>\n") + len("<|im_end|>\n")
        user_end = find_last_user_message_end(messages, tokenizer, enable_thinking, tools)
        out.append({"role": "system", "content": full[:system_end]})
        out.append({"role": "user", "content": full[system_end:user_end]})
        prev = user_end
        begin = max(i for i, m in enumerate(messages) if m.get("role") == "user") + 1
    else:
        prev = 0
        begin = 0

    for i in range(begin, len(messages)):
        if i + 1 < len(messages) and messages[i].get("role") == "tool" and messages[i + 1].get("role") == "tool":
            continue

        add_generation_prompt = messages[i].get("role") in ("tool", "user")
        if enable_thinking and messages[i].get("role") != "assistant":
            nxt = messages[i + 1] if i + 1 < len(messages) else {}
            if nxt.get("role") == "assistant" and not nxt.get("reasoning_content"):
                part = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=tools,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                )
                part += "<|im_start|>assistant\n<think></think>"
            else:
                part = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    tools=tools,
                    chat_template_kwargs={"enable_thinking": enable_thinking},
                )
        else:
            part = tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )

        cur = len(part)
        if part != full[:cur]:
            raise ValueError(f"Template mismatch at message index {i}")
        out.append({"role": messages[i]["role"], "content": full[prev:cur]})
        prev = cur
    return out


def create_masked_messages(
    messages: List[Dict[str, Any]],
    tokenizer,
    tools: Optional[Any],
    start_from_last_user: bool,
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, Any]]]]:
    """Create one or multiple chunk groups for a conversation."""
    has_thinking = any(msg.get("reasoning_content") for msg in messages)
    if not has_thinking:
        chunks = split_template_into_messages(
            messages, tokenizer, start_from_last_user=False, enable_thinking=False, tools=tools
        )
        return [(chunks, messages)]

    user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    result = []
    for idx, uidx in enumerate(user_idxs):
        msgs = messages if idx == len(user_idxs) - 1 else messages[: user_idxs[idx + 1]]
        msgs = deepcopy(msgs)
        last_user_idx = max(i for i, m in enumerate(msgs) if m.get("role") == "user")
        for i in range(last_user_idx + 1):
            if msgs[i].get("reasoning_content"):
                msgs[i]["reasoning_content"] = ""
        chunks = split_template_into_messages(
            msgs,
            tokenizer,
            start_from_last_user=start_from_last_user,
            enable_thinking=True,
            tools=tools,
        )
        result.append((chunks, msgs))
    return result


def _tool_validation(messages: List[Dict[str, Any]]) -> Optional[str]:
    any_tool_call = any(
        ("<tool_call>" in (m.get("content") or "")) or ("<tool_call>" in (m.get("reasoning_content") or ""))
        for m in messages
        if isinstance(m, dict)
    )
    if not any_tool_call:
        return None
    any_tools_header = any(
        ("# Tools" in (m.get("content") or "")) or ("# Tools" in (m.get("reasoning_content") or ""))
        for m in messages
        if isinstance(m, dict)
    )
    if not any_tools_header:
        return "tool_call present but # Tools missing"
    return None


def process_record(
    data: Dict[str, Any],
    tokenizer,
    include_extra: bool,
    start_from_last_user: bool,
) -> List[Dict[str, Any]]:
    messages = replace_json_args(data.get("messages", []))
    if not messages:
        return []
    err = _tool_validation(messages)
    if err:
        return []

    tools = data.get("tools")
    groups = create_masked_messages(messages, tokenizer, tools, start_from_last_user)
    output: List[Dict[str, Any]] = []

    for chunks, msgs in groups:
        rendered = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs={"enable_thinking": any(m.get("reasoning_content") for m in msgs)},
        )
        joined = "".join(x["content"] for x in chunks)
        if rendered != joined:
            continue
        if "<tool_call>" in rendered and "# Tools" not in rendered:
            continue

        token_count = len(tokenizer.encode(joined, add_special_tokens=False))
        record = {"messages": chunks, "num_generated_tokens": token_count}
        if include_extra:
            for k, v in data.items():
                if k != "messages":
                    record[k] = v
        output.append(record)
    return output


def _process_task_mp(task: Tuple[int, str]):
    line_no, line = task
    try:
        data = _loads(line)
    except Exception:
        return line_no, [], line
    try:
        out_items = process_record(
            data,
            _WORKER_TOKENIZER,
            _WORKER_INCLUDE_EXTRA,
            _WORKER_START_FROM_LAST_USER,
        )
        out_lines = [_dumps(x) for x in out_items]
        if out_lines:
            return line_no, out_lines, None
        return line_no, [], _dumps(data)
    except Exception:
        return line_no, [], line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw JSONL path.")
    parser.add_argument("--output", required=True, help="Materialized output JSONL.")
    parser.add_argument("--tokenizer-model", required=True, help="HF tokenizer model/path.")
    parser.add_argument("--chat-template-file", default=None, help="Optional jinja template file.")
    parser.add_argument("--extra-info", action="store_true", help="Keep original extra fields.")
    parser.add_argument("--start-from-last-user", action="store_true", default=True)
    parser.add_argument("--debug-failures", default=None, help="Optional failed-record jsonl.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--chunksize", type=int, default=8, help="Pool imap chunksize.")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=args.trust_remote_code)
    chat_template_text = None
    if args.chat_template_file:
        chat_template_text = Path(args.chat_template_file).read_text(encoding="utf-8")
        tok.chat_template = chat_template_text

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dbg_path = Path(args.debug_failures) if args.debug_failures else None

    total = 0
    kept = 0

    if args.workers <= 1:
        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="materialize"):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    data = _loads(line)
                    out_items = process_record(data, tok, args.extra_info, args.start_from_last_user)
                    for item in out_items:
                        fout.write(_dumps(item) + "\n")
                        kept += 1
                    if not out_items and dbg_path:
                        dbg_path.parent.mkdir(parents=True, exist_ok=True)
                        with dbg_path.open("a", encoding="utf-8") as dbg:
                            dbg.write(_dumps(data) + "\n")
                except Exception:
                    if dbg_path:
                        dbg_path.parent.mkdir(parents=True, exist_ok=True)
                        with dbg_path.open("a", encoding="utf-8") as dbg:
                            dbg.write(line + "\n")
    else:
        def _tasks():
            with in_path.open("r", encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    yield (line_no, line)

        with out_path.open("w", encoding="utf-8") as fout, mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                args.tokenizer_model,
                chat_template_text,
                args.trust_remote_code,
                args.extra_info,
                args.start_from_last_user,
            ),
        ) as pool:
            iterator = pool.imap_unordered(_process_task_mp, _tasks(), chunksize=max(1, args.chunksize))
            for _, out_lines, failed_raw in tqdm(iterator, desc="materialize(mp)"):
                total += 1
                for s in out_lines:
                    fout.write(s + "\n")
                    kept += 1
                if not out_lines and dbg_path and failed_raw is not None:
                    dbg_path.parent.mkdir(parents=True, exist_ok=True)
                    with dbg_path.open("a", encoding="utf-8") as dbg:
                        dbg.write(failed_raw + "\n")

    print(f"materialize done: kept={kept} dropped={total-kept} total={total} output={out_path}")


if __name__ == "__main__":
    main()
