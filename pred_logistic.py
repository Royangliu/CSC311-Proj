import csv

_CLASS_NAMES = (
    "The Persistence of Memory",
    "The Starry Night",
    "The Water Lily Pond",
)

# bias 
_INTERCEPTS = (
    -0.21769237586011414,
    1.0406204056698005,
    -0.8229280298096902,
)

# weights per feature for each class
_COEF = (
    (
        1.0612962772252243,
        -0.8874906709599167,
        -0.20281414738702308,
        0.8722263691059515,
    ),
    (
        0.262939906796793,
        0.0069636996163808295,
        -0.07625964670339269,
        -0.14483327655527925,
    ),
    (
        -1.3242361840220152,
        0.8805269713435377,
        0.2790737940904134,
        -0.7273930925506702,
    ),
)

_FEAT_COLS = (
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
)


def _cell_float(value):
    if value is None:
        return 0.0
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def predict(row):
    """
    One row: mapping-like object (e.g. dict from csv.DictReader) keyed by CSV header.
    Missing / non-numeric emotion fields are treated as 0.0.
    """
    x = [_cell_float(row.get(name)) for name in _FEAT_COLS]
    best_k = 0
    best_score = None
    for k in range(3):
        s = _INTERCEPTS[k]
        row_coef = _COEF[k]
        s += row_coef[0] * x[0]
        s += row_coef[1] * x[1]
        s += row_coef[2] * x[2]
        s += row_coef[3] * x[3]
        if best_score is None or s > best_score:
            best_score = s
            best_k = k
    return _CLASS_NAMES[best_k]


def predict_all(filename):
    """
    Read `filename` as UTF-8 CSV with headers; return predictions in row order.
    """
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out = []
    for row in rows:
        out.append(predict(row))
    return out
