"""
Append bag-of-words indicator columns from bagofwords*.txt to train/val/test_norm.csv.

Feel vocabulary -> match in 'Describe how this painting makes you feel.'
Food vocabulary -> match in 'If this painting was a food, what would be?'
Music vocabulary -> match in soundtrack free-text column.

Writes:
  train_norm.csv, val_norm.csv, test_norm.csv (overwrites in this directory)
  validation_norm.csv (same contents as val_norm.csv)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

COL_FEEL = "Describe how this painting makes you feel."
COL_FOOD = "If this painting was a food, what would be?"
COL_SOUNDTRACK = (
    "Imagine a soundtrack for this painting. Describe that soundtrack "
    "without naming any objects in the painting."
)


def _load_vocab_feel_food(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [t for t in text.split() if t.strip()]


def _load_vocab_music(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split(",") if p.strip()]


def _ascii_word_token(tok: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9']+", tok, flags=re.IGNORECASE))


def _unique_column_names(prefix: str, raw_tokens: list[str]) -> list[tuple[str, str]]:
    """Return (raw_token, column_name) with stable unique names."""
    out: list[tuple[str, str]] = []
    counts: dict[str, int] = {}

    for i, raw in enumerate(raw_tokens):
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip()).strip("_")
        if not slug:
            slug = f"t{i}"
        slug = slug[:100]
        base = f"{prefix}{slug}"
        n = counts.get(base, 0)
        col = f"{base}__{n}" if n else base
        counts[base] = n + 1
        out.append((raw, col))
    return out


def _bow_frame(series: pd.Series, mapping: list[tuple[str, str]]) -> pd.DataFrame:
    """Binary indicators for each raw token against series text (vectorized)."""
    lowered = series.fillna("").astype(str).str.lower()
    cols: dict[str, pd.Series] = {}
    for raw, col in mapping:
        tok = raw.strip().lower()
        if not tok:
            cols[col] = pd.Series(0, index=series.index, dtype="int8")
            continue
        if _ascii_word_token(tok):
            pat = r"\b" + re.escape(tok) + r"\b"
            cols[col] = lowered.str.contains(pat, regex=True, na=False).astype("int8")
        else:
            cols[col] = lowered.str.contains(re.escape(tok), regex=True, na=False).astype(
                "int8"
            )
    return pd.DataFrame(cols)


def _strip_placeholder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty pandas 'Unnamed: N' columns from malformed CSV headers."""
    drop = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=drop, errors="ignore")


def main() -> None:
    feel_tokens = _load_vocab_feel_food(HERE / "bagofwordsfeel.txt")
    food_tokens = _load_vocab_feel_food(HERE / "bagofwordsfood.txt")
    music_tokens = _load_vocab_music(HERE / "bagofwordsmusic.txt")

    feel_map = _unique_column_names("bow_feel_", feel_tokens)
    food_map = _unique_column_names("bow_food_", food_tokens)
    music_map = _unique_column_names("bow_music_", music_tokens)

    splits = {
        "train_norm.csv": HERE / "train_norm.csv",
        "val_norm.csv": HERE / "val_norm.csv",
        "test_norm.csv": HERE / "test_norm.csv",
    }

    for name, path in splits.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        df = _strip_placeholder_columns(pd.read_csv(path))
        bow_prefixes = ("bow_feel_", "bow_food_", "bow_music_")
        extra = [c for c in df.columns if c.startswith(bow_prefixes)]
        if extra:
            df = df.drop(columns=extra)
        for col in (COL_FEEL, COL_FOOD, COL_SOUNDTRACK):
            if col not in df.columns:
                raise KeyError(f"Column not found: {col!r} in {name}")

        bow = pd.concat(
            [
                _bow_frame(df[COL_FEEL], feel_map),
                _bow_frame(df[COL_FOOD], food_map),
                _bow_frame(df[COL_SOUNDTRACK], music_map),
            ],
            axis=1,
        )
        out = pd.concat([df, bow], axis=1)
        out.to_csv(path, index=False)
        print(f"Wrote {path.name}: +{bow.shape[1]} BOW columns -> {out.shape[1]} total columns")

    val_path = HERE / "val_norm.csv"
    validation_path = HERE / "validation_norm.csv"
    validation_path.write_bytes(val_path.read_bytes())
    print(f"Wrote validation_norm.csv (copy of val_norm.csv)")


if __name__ == "__main__":
    main()
