from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from transformers import T5TokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute T5 tokenizer statistics for Q4.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing train/dev .nl and .sql files.",
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        default="",
        help="Optional prefix to prepend to NL questions in the processed view (e.g., 'translate to sql:').",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase NL and SQL strings for the processed view.",
    )
    parser.add_argument(
        "--strip-schema",
        action="store_true",
        help="Remove schema annotations in square brackets from NL strings for the processed view.",
    )
    parser.add_argument(
        "--max-nl-chars",
        type=int,
        default=0,
        help="If > 0, truncate processed NL strings to this many characters.",
    )
    parser.add_argument(
        "--max-sql-chars",
        type=int,
        default=0,
        help="If > 0, truncate processed SQL strings to this many characters.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="google-t5/t5-small",
        help="Tokenizer checkpoint to use for statistics.",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for tokenizer files (if already downloaded).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local files when loading the tokenizer (set this if running offline).",
    )
    return parser.parse_args()


def load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


def strip_schema_tokens(sentence: str) -> str:
    """Remove bracketed schema hints like '[table.column]'."""
    output_chars: List[str] = []
    skip = False
    for char in sentence:
        if char == "[":
            skip = True
            continue
        if char == "]":
            skip = False
            continue
        if not skip:
            output_chars.append(char)
    return "".join(output_chars)


def preprocess_strings(
    strings: Sequence[str],
    prefix: str = "",
    lowercase: bool = False,
    strip_schema: bool = False,
    max_chars: int = 0,
) -> List[str]:
    processed: List[str] = []
    for text in strings:
        new_text = text.strip()
        if lowercase:
            new_text = new_text.lower()
        if strip_schema:
            new_text = strip_schema_tokens(new_text)
        if prefix:
            new_text = f"{prefix.strip()} {new_text}"
        if max_chars > 0:
            new_text = new_text[:max_chars]
        processed.append(new_text.strip())
    return processed


def tokenize_sequences(tokenizer: T5TokenizerFast, sequences: Sequence[str]) -> List[List[int]]:
    encoded = tokenizer(
        list(sequences),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return encoded["input_ids"]


def compute_stats(
    tokenizer: T5TokenizerFast, nl_strings: Sequence[str], sql_strings: Sequence[str]
) -> Dict[str, float]:
    nl_tokens = tokenize_sequences(tokenizer, nl_strings)
    sql_tokens = tokenize_sequences(tokenizer, sql_strings)

    def summarize(token_lists: Sequence[Sequence[int]]) -> Tuple[float, int]:
        lengths = [len(seq) for seq in token_lists]
        vocab = set(token for seq in token_lists for token in seq)
        mean_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        return mean_len, len(vocab)

    nl_mean, nl_vocab = summarize(nl_tokens)
    sql_mean, sql_vocab = summarize(sql_tokens)

    return {
        "num_examples": len(nl_strings),
        "mean_nl_len": nl_mean,
        "mean_sql_len": sql_mean,
        "nl_vocab": nl_vocab,
        "sql_vocab": sql_vocab,
    }


def describe_split(
    split_name: str,
    tokenizer: T5TokenizerFast,
    nl_raw: Sequence[str],
    sql_raw: Sequence[str],
    nl_processed: Sequence[str],
    sql_processed: Sequence[str],
) -> None:
    print(f"\n=== {split_name.upper()} SPLIT ===")
    raw_stats = compute_stats(tokenizer, nl_raw, sql_raw)
    proc_stats = compute_stats(tokenizer, nl_processed, sql_processed)

    def fmt(stats: Dict[str, float]) -> str:
        return (
            f"#examples={stats['num_examples']}, "
            f"mean_NL_len={stats['mean_nl_len']:.2f}, "
            f"mean_SQL_len={stats['mean_sql_len']:.2f}, "
            f"NL_vocab={stats['nl_vocab']}, "
            f"SQL_vocab={stats['sql_vocab']}"
        )

    print("Before preprocessing:", fmt(raw_stats))
    print("After preprocessing :", fmt(proc_stats))


def main() -> None:
    args = parse_args()
    try:
        tokenizer = T5TokenizerFast.from_pretrained(
            args.tokenizer_name,
            cache_dir=str(args.tokenizer_cache_dir) if args.tokenizer_cache_dir else None,
            local_files_only=args.local_files_only,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "Failed to load tokenizer. Download 'google-t5/t5-small' first, provide --tokenizer-cache-dir "
            "to a local copy, or run without --local-files-only so the files can be fetched."
        ) from exc

    for split in ("train", "dev"):
        nl_path = args.data_dir / f"{split}.nl"
        sql_path = args.data_dir / f"{split}.sql"
        if not nl_path.exists() or not sql_path.exists():
            raise FileNotFoundError(f"Missing files for split '{split}' in {args.data_dir}")

        nl_raw = load_lines(nl_path)
        sql_raw = load_lines(sql_path)
        if len(nl_raw) != len(sql_raw):
            raise ValueError(f"Mismatch between NL and SQL counts in {split} split.")

        nl_processed = preprocess_strings(
            nl_raw,
            prefix=args.task_prefix,
            lowercase=args.lowercase,
            strip_schema=args.strip_schema,
            max_chars=args.max_nl_chars,
        )
        sql_processed = preprocess_strings(
            sql_raw,
            lowercase=args.lowercase,
            max_chars=args.max_sql_chars,
        )

        describe_split(
            split,
            tokenizer,
            nl_raw,
            sql_raw,
            nl_processed,
            sql_processed,
        )


if __name__ == "__main__":
    main()
