#!/usr/bin/env python3
"""Convert OpenResearcher parquet rows to example-style JSONL."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_template_tools(template_jsonl: Path) -> list[dict[str, Any]]:
    """Load `tools` field from the first line of a template JSONL file."""
    with template_jsonl.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return []
    obj = json.loads(first_line)
    tools = obj.get("tools", [])
    return tools if isinstance(tools, list) else []


def _extract_text(content: Any) -> str:
    """Extract plain text from source message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "\n".join(parts)
    return ""


def _new_message(
    role: str,
    content: str,
    reasoning_content: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build target-format message object."""
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "role": role,
        "tool_call_id": tool_call_id,
        "tool_calls": tool_calls,
    }


def convert_messages_to_target_format(
    src_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert Cursor-style messages to target OpenAI-style messages.

    Source format has `channel/recipient/content_type`; target uses
    `content/reasoning_content/tool_calls/tool_call_id`.
    """
    out_messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None
    pending_tool_call_ids: deque[str] = deque()
    call_idx = 0

    # Merge system + developer prompt into one `system` message.
    system_parts: list[str] = []
    for msg in src_messages:
        if msg.get("role") in {"system", "developer"}:
            text = _extract_text(msg.get("content"))
            if text.strip():
                system_parts.append(text)
    if system_parts:
        out_messages.append(
            _new_message(
                role="system",
                content="\n\n".join(system_parts),
                reasoning_content=None,
                tool_call_id=None,
                tool_calls=None,
            )
        )

    for msg in src_messages:
        role = msg.get("role")
        text = _extract_text(msg.get("content"))

        if role in {"system", "developer"}:
            continue

        if role == "user":
            out_messages.append(
                _new_message(
                    role="user",
                    content=text,
                    reasoning_content=None,
                    tool_call_id=None,
                    tool_calls=None,
                )
            )
            continue

        if role == "assistant":
            recipient = msg.get("recipient")
            channel = msg.get("channel")

            # Tool invocation: map to assistant message with tool_calls.
            if isinstance(recipient, str) and recipient:
                call_idx += 1
                call_id = f"call_{call_idx:08d}"
                pending_tool_call_ids.append(call_id)
                arguments = text if text else "{}"
                reasoning = pending_reasoning if pending_reasoning else None
                pending_reasoning = None

                out_messages.append(
                    _new_message(
                        role="assistant",
                        content="",
                        reasoning_content=reasoning,
                        tool_call_id=None,
                        tool_calls=[
                            {
                                "function": {
                                    "arguments": arguments,
                                    "name": recipient,
                                },
                                "id": call_id,
                                "type": "function",
                            }
                        ],
                    )
                )
                continue

            # Analysis messages are usually reasoning for the next tool call/final answer.
            if channel == "analysis":
                pending_reasoning = text if text else pending_reasoning
                continue

            # Final assistant response (no tool call).
            out_messages.append(
                _new_message(
                    role="assistant",
                    content=text,
                    reasoning_content=pending_reasoning,
                    tool_call_id=None,
                    tool_calls=None,
                )
            )
            pending_reasoning = None
            continue

        if role == "tool":
            tool_call_id = pending_tool_call_ids.popleft() if pending_tool_call_ids else None
            out_messages.append(
                _new_message(
                    role="tool",
                    content=text,
                    reasoning_content=None,
                    tool_call_id=tool_call_id,
                    tool_calls=None,
                )
            )
            continue

    return out_messages


def convert_parquet_to_jsonl(
    parquet_path: Path,
    output_jsonl_path: Path,
    output_pretty_json_path: Path | None,
    template_jsonl_path: Path | None,
    default_correct: bool,
    batch_size: int = 128,
) -> int:
    """Convert parquet records to JSONL and optional pretty JSON array."""
    template_tools: list[dict[str, Any]] = []
    if template_jsonl_path is not None:
        template_tools = load_template_tools(template_jsonl_path)

    parquet = pq.ParquetFile(parquet_path)
    total_rows = 0

    with output_jsonl_path.open("w", encoding="utf-8") as out_jsonl:
        out_pretty = None
        if output_pretty_json_path is not None:
            out_pretty = output_pretty_json_path.open("w", encoding="utf-8")
            out_pretty.write("[\n")

        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                record = dict(row)
                record["messages"] = convert_messages_to_target_format(
                    record.get("messages", [])
                )
                record.setdefault("correct", default_correct)
                record.setdefault("tools", template_tools)
                out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

                if out_pretty is not None:
                    if total_rows > 0:
                        out_pretty.write(",\n")
                    out_pretty.write(json.dumps(record, ensure_ascii=False, indent=2))

                total_rows += 1

        if out_pretty is not None:
            out_pretty.write("\n]\n")
            out_pretty.close()

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert parquet to example-style JSONL."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            "./OpenResearcher-Dataset/seed_42/train-00000-of-00003.parquet"
        ),
        help="Input parquet file path.",
    )
    parser.add_argument(
        "--template-jsonl",
        type=Path,
        default=Path(
            "./data/converted_gpt_oss_search_correct.example.jsonl"
        ),
        help="Template JSONL (first line used to copy `tools` structure).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "./data/converted_gpt_oss_search_correct.materialized.jsonl"
        ),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--output-pretty-json",
        type=Path,
        default=Path(
            "./data/converted_gpt_oss_search_correct.materialized.pretty.json"
        ),
        help="Pretty-printed JSON array output path.",
    )
    parser.add_argument(
        "--default-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default value for `correct` when source records do not have it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Parquet reading batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_parquet_to_jsonl(
        parquet_path=args.parquet,
        output_jsonl_path=args.output_jsonl,
        output_pretty_json_path=args.output_pretty_json,
        template_jsonl_path=args.template_jsonl,
        default_correct=args.default_correct,
        batch_size=args.batch_size,
    )
    if args.output_pretty_json is not None:
        print(
            f"Done. Wrote {rows} rows to: {args.output_jsonl} "
            f"and {args.output_pretty_json}"
        )
    else:
        print(f"Done. Wrote {rows} rows to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
