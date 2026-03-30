import csv
import os
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

INPUT_CSV_DEFAULT = os.path.join(DATA_DIR, "training_data_202601_val.csv")
BOW_FEEL_PATH = os.path.join(DATA_DIR, "bagofwordsfeel.txt")
BOW_FOOD_PATH = os.path.join(DATA_DIR, "bagofwordsfood.txt")
BOW_MUSIC_PATH = os.path.join(DATA_DIR, "bagofwordsmusic.txt")

COL_FEEL = "Describe how this painting makes you feel."
COL_FOOD = "If this painting was a food, what would be?"
COL_MUSIC = (
    "Imagine a soundtrack for this painting. Describe that soundtrack "
    "without naming any objects in the painting."
)


def process_bow_features(csv_path=INPUT_CSV_DEFAULT):
    """
    1) load vocab from the text files
    2) read CSV,
    3) convert text into BOW features,
    4) add BOW features to each original row.
    Returns list[dict]: one combined row dict per row
    """
    # Unified bag-of-words format: one token/phrase per line.
    with open(BOW_FEEL_PATH, "r", encoding="utf-8") as f:
        feel_vocab = [x.strip() for x in f if x.strip()]
    with open(BOW_FOOD_PATH, "r", encoding="utf-8") as f:
        food_vocab = [x.strip() for x in f if x.strip()]
    with open(BOW_MUSIC_PATH, "r", encoding="utf-8") as f:
        music_vocab = [x.strip() for x in f if x.strip()]

    feel_vocab = list(dict.fromkeys(feel_vocab))
    food_vocab = list(dict.fromkeys(food_vocab))
    music_vocab = list(dict.fromkeys(music_vocab))

    def has_token(text, token):
        t = token.lower().strip()
        if not t:
            return False
        lower_text = (text or "").lower()
        if re.fullmatch(r"[a-z0-9']+", t):
            return re.search(r"\b" + re.escape(t) + r"\b", lower_text) is not None
        return t in lower_text

    def feature_name(prefix, token, i):
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", token.strip()).strip("_")
        if not slug:
            slug = f"tok_{i}"
        return f"{prefix}{slug[:100]}"

    rows_with_features = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combined = dict(row)
            feel_text = row.get(COL_FEEL, "")
            food_text = row.get(COL_FOOD, "")
            music_text = row.get(COL_MUSIC, "")

            for i, tok in enumerate(feel_vocab):
                combined[feature_name("bow_feel_", tok, i)] = int(has_token(feel_text, tok))
            for i, tok in enumerate(food_vocab):
                combined[feature_name("bow_food_", tok, i)] = int(has_token(food_text, tok))
            for i, tok in enumerate(music_vocab):
                combined[feature_name("bow_music_", tok, i)] = int(has_token(music_text, tok))

            rows_with_features.append(combined)

    return rows_with_features


def predict(_row):
    pass

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    rows = process_bow_features(filename)
    predictions = []
    for row in rows:
        pred = predict(row)
        predictions.append(pred)

    return predictions
