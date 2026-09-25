import io
import re
import hashlib
from difflib import SequenceMatcher

import streamlit as st
import fitz
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ Defence News Scanner OCR")
st.caption(
    "Layout-aware newspaper OCR for complete defence-related news articles"
)


# ============================================================
# DEFENCE VOCABULARY
# ============================================================

# Very strong signals.
# These indicate that the article itself is likely about defence/military affairs.

VERY_STRONG_PHRASES = [
    "indian army",
    "indian navy",
    "indian air force",
    "armed forces",
    "ministry of defence",
    "ministry of defense",
    "defence ministry",
    "defense ministry",
    "chief of defence staff",
    "chief of defense staff",
    "air defence",
    "air defense",
    "missile defence",
    "missile defense",
    "air strike",
    "airstrike",
    "air strikes",
    "naval operation",
    "military operation",
    "military operations",
    "military exercise",
    "military exercises",
    "military commander",
    "corps commander",
    "corps commanders",
    "army chief",
    "navy chief",
    "air force chief",
    "army personnel",
    "military personnel",
    "military deployment",
    "troop deployment",
    "troops deployed",
    "defence procurement",
    "defense procurement",
    "defence deal",
    "defense deal",
    "defence equipment",
    "defense equipment",
    "fighter aircraft",
    "fighter jet",
    "fighter jets",
    "warship",
    "warships",
    "submarine",
    "submarines",
    "ballistic missile",
    "ballistic missiles",
    "cruise missile",
    "cruise missiles",
    "surface-to-air missile",
    "surface to air missile",
    "anti-aircraft",
    "anti aircraft",
    "special forces",
    "paramilitary forces",
    "border security",
    "border forces",
    "line of actual control",
    "lac",
    "line of control",
    "loc",
    "military talks",
    "defence talks",
    "defense talks",
    "security talks",
    "military standoff",
    "military tensions",
    "military confrontation",
    "military conflict",
    "armed conflict",
    "ceasefire",
    "military strike",
    "military strikes",
    "missile attack",
    "missile attacks",
    "drone attack",
    "drone attacks",
    "drone strike",
    "drone strikes",
    "artillery fire",
    "artillery shelling",
    "rocket attack",
    "rocket attacks",
    "military aircraft",
    "military helicopter",
    "military helicopters",
    "military vessel",
    "military vessels",
    "air force",
    "naval forces",
    "ground forces",
    "defence forces",
    "defense forces",
]


# Supporting terms.
# These are NEVER sufficient by themselves.

STRONG_TERMS = [
    "army",
    "navy",
    "air force",
    "military",
    "soldier",
    "soldiers",
    "troop",
    "troops",
    "commander",
    "commanders",
    "brigade",
    "regiment",
    "battalion",
    "corps",
    "missile",
    "missiles",
    "fighter",
    "fighters",
    "helicopter",
    "helicopters",
    "tank",
    "tanks",
    "drone",
    "drones",
    "warship",
    "submarine",
    "weapons",
    "weapon",
    "ammunition",
    "artillery",
    "border",
    "frontier",
    "airbase",
    "air base",
    "naval",
    "aircraft",
    "defence",
    "defense",
    "security forces",
    "special forces",
    "operation",
    "operations",
    "strike",
    "strikes",
    "attack",
    "attacks",
    "ceasefire",
    "conflict",
    "war",
    "combat",
    "deployment",
    "deployed",
]


# Context words.
# These alone MUST NOT qualify an article.

CONTEXT_WORDS = [
    "russia",
    "ukraine",
    "iran",
    "israel",
    "china",
    "pakistan",
    "lebanon",
    "gaza",
    "syria",
    "us",
    "usa",
    "united states",
    "moscow",
    "tehran",
    "kyiv",
    "beijing",
]


# Terms that frequently create false positives.

NON_DEFENCE_TOPICS = [
    "job",
    "jobs",
    "employment",
    "unemployment",
    "graduate",
    "graduates",
    "university",
    "college",
    "school",
    "education",
    "dance",
    "dancer",
    "bharatanatyam",
    "music",
    "film",
    "actor",
    "actress",
    "culture",
    "cultural",
    "festival",
    "fashion",
    "food",
    "restaurant",
    "business",
    "stock",
    "stocks",
    "market",
    "economy",
    "economic",
    "inflation",
    "volcano",
    "earthquake",
    "flood",
    "hostel",
    "building collapse",
    "collapse",
    "rescue",
    "survivor",
    "football",
    "cricket",
    "tennis",
    "sports",
]


# ============================================================
# TEXT NORMALISATION
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    # OCR junk
    text = text.replace("\x0c", " ")
    text = text.replace("|", " ")
    text = text.replace("~", " ")
    text = text.replace("_", " ")

    # Join words split by line hyphenation
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Fix spaces around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Newline cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalise_for_matching(text):
    text = text.lower()

    replacements = {
        "defence": "defense",
        "organisation": "organization",
        "organised": "organized",
        "programme": "program",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# DEFENCE CLASSIFIER
# ============================================================

def count_phrase_hits(text, phrases):
    low = normalise_for_matching(text)

    found = []

    for phrase in phrases:
        p = normalise_for_matching(phrase)

        if p in low:
            found.append(phrase)

    return found


def count_term_hits(text, terms):
    low = normalise_for_matching(text)

    found = []

    for term in terms:
        p = normalise_for_matching(term)

        # word-ish matching
        if re.search(r"\b" + re.escape(p) + r"\b", low):
            found.append(term)

    return found


def defence_analysis(text):
    """
    Strict article-level defence classification.

    Important:
    Country names alone do NOT qualify.
    One weak word like 'army' does NOT qualify.
    A cultural/business/rescue article containing 'army' is rejected.
    """

    if not text:
        return {
            "is_defence": False,
            "score": 0,
            "signals": [],
            "reason": "empty"
        }

    cleaned = clean_text(text)
    low = normalise_for_matching(cleaned)

    words = re.findall(r"\b[a-zA-Z]{2,}\b", low)

    if len(words) < 35:
        return {
            "is_defence": False,
            "score": 0,
            "signals": [],
            "reason": "too_short"
        }

    very_strong = count_phrase_hits(
        cleaned,
        VERY_STRONG_PHRASES
    )

    strong = count_term_hits(
        cleaned,
        STRONG_TERMS
    )

    context = count_term_hits(
        cleaned,
        CONTEXT_WORDS
    )

    non_defence = count_term_hits(
        cleaned,
        NON_DEFENCE_TOPICS
    )

    score = 0
    signals = []

    # Very strong phrases carry substantial weight.
    for item in very_strong:
        score += 4
        signals.append(item)

    # Supporting defence vocabulary.
    unique_strong = list(dict.fromkeys(strong))

    for item in unique_strong:
        score += 1
        signals.append(item)

    # Context is only useful when actual military vocabulary exists.
    if context and (very_strong or len(unique_strong) >= 2):
        score += min(len(context), 3)

    # Penalise obvious non-defence stories.
    for item in non_defence:
        score -= 1

    # Special rejection:
    # If only a single weak military word appears in a cultural/business/etc article,
    # reject it.

    if (
        not very_strong
        and len(unique_strong) <= 2
        and len(non_defence) >= 2
    ):
        return {
            "is_defence": False,
            "score": score,
            "signals": list(dict.fromkeys(signals)),
            "reason": "non_defence_context"
        }

    # Need either:
    # - at least one very strong defence phrase
    # - OR several strong defence terms
    #
    # And preferably enough total evidence.

    if very_strong and score >= 5:
        return {
            "is_defence": True,
            "score": score,
            "signals": list(dict.fromkeys(signals)),
            "reason": "strong_phrase"
        }

    if len(unique_strong) >= 4 and score >= 6:
        return {
            "is_defence": True,
            "score": score,
            "signals": list(dict.fromkeys(signals)),
            "reason": "multiple_military_terms"
        }

    if len(unique_strong) >= 3 and len(context) >= 1 and score >= 6:
        return {
            "is_defence": True,
            "score": score,
            "signals": list(dict.fromkeys(signals)),
            "reason": "military_context"
        }

    return {
        "is_defence": False,
        "score": score,
        "signals": list(dict.fromkeys(signals)),
        "reason": "insufficient_defence_evidence"
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    image = image.convert("L")

    # Upscale
    width, height = image.size

    target_width = max(width, 2400)

    if width < target_width:
        ratio = target_width / width
        image = image.resize(
            (
                int(width * ratio),
                int(height * ratio)
            ),
            Image.Resampling.LANCZOS
        )

    # Contrast
    image = ImageEnhance.Contrast(image).enhance(1.6)

    # Sharpness
    image = ImageEnhance.Sharpness(image).enhance(1.5)

    # Light denoise
    image = image.filter(ImageFilter.MedianFilter(size=3))

    return image


# ============================================================
# PDF RENDER
# ============================================================

def render_pdf_page(page, zoom=2.4):

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    img = Image.open(
        io.BytesIO(pix.tobytes("png"))
    )

    return img


# ============================================================
# OCR DATA
# ============================================================

def get_ocr_data(image, psm=6):

    config = (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1"
    )

    try:
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config=config,
            lang="eng"
        )
    except Exception:
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config=config
        )

    return data


# ============================================================
# FIND NEWSPAPER COLUMNS
# ============================================================

def detect_columns(image):
    """
    Detect columns from OCR word positions.

    This is intentionally based on horizontal distribution,
    not vertical sorting.
    """

    data = get_ocr_data(image, psm=3)

    width, height = image.size

    words = []

    n = len(data["text"])

    for i in range(n):

        text = data["text"][i].strip()

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1

        if not text:
            continue

        if conf < 20:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        if w <= 2 or h <= 2:
            continue

        words.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + w / 2,
                "text": text
            }
        )

    if len(words) < 30:
        return [(0, width)]

    # Build horizontal histogram.
    bins = 100
    hist = [0] * bins

    for word in words:

        cx = word["cx"]

        idx = int((cx / width) * bins)

        idx = max(0, min(bins - 1, idx))

        hist[idx] += 1

    # Smooth histogram.
    smooth = hist[:]

    for i in range(2, bins - 2):
        smooth[i] = (
            hist[i - 2]
            + hist[i - 1]
            + hist[i]
            + hist[i + 1]
            + hist[i + 2]
        )

    # Find deep vertical gaps.
    candidates = []

    max_hist = max(smooth) if smooth else 1

    for i in range(3, bins - 3):

        left = sum(smooth[max(0, i - 3):i])
        right = sum(smooth[i + 1:min(bins, i + 4)])

        center = smooth[i]

        if center <= max_hist * 0.08 and left > 10 and right > 10:
            x = int(width * i / bins)

            # Avoid edges
            if 0.12 * width < x < 0.88 * width:
                candidates.append(x)

    # Remove candidates too close to each other.
    filtered = []

    for x in candidates:

        if not filtered or x - filtered[-1] > width * 0.12:
            filtered.append(x)

    # We normally want max 4 columns.
    if len(filtered) > 3:

        # Choose deepest gaps.
        scored = []

        for x in filtered:

            idx = int((x / width) * bins)

            score = smooth[
                max(0, idx - 1):
                min(bins, idx + 2)
            ]

            scored.append(
                (
                    sum(score),
                    x
                )
            )

        scored.sort()

        filtered = sorted(
            [x for _, x in scored[:3]]
        )

    boundaries = [0] + filtered + [width]

    columns = []

    for i in range(len(boundaries) - 1):

        left = boundaries[i]
        right = boundaries[i + 1]

        # Small margin so text does not touch neighbouring column.
        margin = int(width * 0.008)

        left = max(0, left + margin)
        right = min(width, right - margin)

        if right - left > width * 0.15:
            columns.append((left, right))

    if not columns:
        return [(0, width)]

    return columns


# ============================================================
# COLUMN CROPPING
# ============================================================

def crop_column(image, bounds):

    left, right = bounds

    width, height = image.size

    # Don't crop vertically.
    return image.crop(
        (
            max(0, left),
            0,
            min(width, right),
            height
        )
    )


# ============================================================
# OCR COLUMN
# ============================================================

def ocr_column(column_image):

    # PSM 4 = single column newspaper text.
    config = (
        "--oem 3 --psm 4 "
        "-c preserve_interword_spaces=1"
    )

    try:
        data = pytesseract.image_to_data(
            column_image,
            output_type=Output.DICT,
            config=config,
            lang="eng"
        )
    except Exception:
        data = pytesseract.image_to_data(
            column_image,
            output_type=Output.DICT,
            config=config
        )

    items = []

    n = len(data["text"])

    for i in range(n):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1

        if conf < 15:
            continue

        item = {
            "text": text,
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
            "conf": conf,
            "block": int(data["block_num"][i]),
            "paragraph": int(data["par_num"][i]),
            "line": int(data["line_num"][i]),
        }

        items.append(item)

    return items


# ============================================================
# RECONSTRUCT OCR LINES
# ============================================================

def build_lines(items):

    groups = {}

    for item in items:

        key = (
            item["block"],
            item["paragraph"],
            item["line"]
        )

        groups.setdefault(key, []).append(item)

    lines = []

    for key, words in groups.items():

        words.sort(key=lambda x: x["x"])

        text = " ".join(
            w["text"]
            for w in words
        )

        text = clean_text(text)

        if not text:
            continue

        x1 = min(w["x"] for w in words)
        y1 = min(w["y"] for w in words)
        x2 = max(w["x"] + w["w"] for w in words)
        y2 = max(w["y"] + w["h"] for w in words)

        height = max(
            w["h"]
            for w in words
        )

        confidence = sum(
            w["conf"]
            for w in words
        ) / len(words)

        lines.append(
            {
                "text": text,
                "x": x1,
                "y": y1,
                "right": x2,
                "bottom": y2,
                "height": height,
                "confidence": confidence,
                "block": key[0],
                "paragraph": key[1],
                "line": key[2],
            }
        )

    lines.sort(
        key=lambda x: (
            x["y"],
            x["x"]
        )
    )

    return lines


# ============================================================
# HEADLINE DETECTION
# ============================================================

def looks_like_headline(line):

    text = line["text"].strip()

    if len(text) < 4:
        return False

    words = text.split()

    if len(words) > 18:
        return False

    # Headlines are usually taller than body text.
    if line["height"] >= 28:
        return True

    # Uppercase headline.
    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if len(letters) >= 8:

        upper_ratio = (
            sum(c.isupper() for c in letters)
            / len(letters)
        )

        if upper_ratio > 0.72 and len(words) <= 14:
            return True

    # Typical headline punctuation.
    if (
        len(words) <= 12
        and (
            ":" in text
            or "?" in text
            or "!" in text
        )
        and line["height"] >= 20
    ):
        return True

    return False


# ============================================================
# FILTER OCR JUNK
# ============================================================

def is_probable_body_line(text):

    text = text.strip()

    if not text:
        return False

    # Ignore tiny page furniture.
    if len(text) <= 2:
        return False

    # Mostly garbage symbols.
    letters = sum(
        c.isalpha()
        for c in text
    )

    digits = sum(
        c.isdigit()
        for c in text
    )

    if letters < 2:
        return False

    if len(text) > 3 and letters / len(text) < 0.25:
        return False

    # Page numbers etc.
    if digits > letters * 3:
        return False

    return True


# ============================================================
# ARTICLE SEGMENTATION
# ============================================================

def split_column_into_articles(lines, column_height):

    """
    Main improvement.

    We do NOT simply dump the whole column into one article.

    Articles are separated using:
      - headline detection
      - large vertical gaps
      - block transitions
      - OCR layout changes
    """

    valid = [
        line for line in lines
        if is_probable_body_line(line["text"])
    ]

    if not valid:
        return []

    # Estimate normal body line spacing.
    gaps = []

    for i in range(1, len(valid)):

        gap = (
            valid[i]["y"]
            - valid[i - 1]["bottom"]
        )

        if 0 <= gap < 300:
            gaps.append(gap)

    if gaps:
        gaps_sorted = sorted(gaps)

        median_gap = gaps_sorted[
            len(gaps_sorted) // 2
        ]
    else:
        median_gap = 8

    # Large gap threshold.
    large_gap = max(
        28,
        median_gap * 3.2
    )

    articles = []

    current = []

    for i, line in enumerate(valid):

        if not current:
            current = [line]
            continue

        previous = current[-1]

        gap = (
            line["y"]
            - previous["bottom"]
        )

        new_headline = looks_like_headline(line)

        previous_was_body = (
            not looks_like_headline(previous)
        )

        # Strong separation.
        if gap >= large_gap:

            if current:
                articles.append(current)

            current = [line]
            continue

        # New large headline after existing content.
        if (
            new_headline
            and len(current) >= 3
            and previous_was_body
            and gap >= median_gap
        ):

            articles.append(current)

            current = [line]
            continue

        current.append(line)

    if current:
        articles.append(current)

    # Remove tiny blocks.
    result = []

    for article in articles:

        text = " ".join(
            line["text"]
            for line in article
        )

        text = clean_text(text)

        word_count = len(
            re.findall(
                r"\b[A-Za-z]{2,}\b",
                text
            )
        )

        if word_count < 25:
            continue

        result.append(article)

    return result


# ============================================================
# ARTICLE TEXT CLEANUP
# ============================================================

def article_to_text(article_lines):

    if not article_lines:
        return ""

    output = []

    for line in article_lines:

        text = clean_text(
            line["text"]
        )

        if not text:
            continue

        output.append(text)

    # Join lines intelligently.
    result = ""

    for line in output:

        if not result:
            result = line
            continue

        # If previous ends with hyphen,
        # join without a space.
        if result.endswith("-"):
            result = result[:-1] + line
        else:
            result += " " + line

    result = clean_text(result)

    # Repair common OCR spacing.
    result = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        result
    )

    return result.strip()


# ============================================================
# ARTICLE QUALITY
# ============================================================

def article_quality(text):

    if not text:
        return False

    words = re.findall(
        r"\b[A-Za-z]{2,}\b",
        text
    )

    if len(words) < 35:
        return False

    # Need reasonable sentence structure.
    sentences = re.split(
        r"[.!?]+",
        text
    )

    meaningful_sentences = [
        s for s in sentences
        if len(s.split()) >= 5
    ]

    if len(meaningful_sentences) < 2:
        return False

    # Too much OCR garbage.
    letters = sum(
        c.isalpha()
        for c in text
    )

    if len(text) > 0:

        ratio = letters / len(text)

        if ratio < 0.45:
            return False

    return True


# ============================================================
# DUPLICATE CHECK
# ============================================================

def text_fingerprint(text):

    normalized = normalise_for_matching(text)

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    )

    return hashlib.md5(
        normalized.encode("utf-8")
    ).hexdigest()


def is_duplicate(text, previous_texts):

    normalized = normalise_for_matching(text)

    for old in previous_texts:

        old_normalized = normalise_for_matching(old)

        # Exact-ish.
        if normalized == old_normalized:
            return True

        # Similarity.
        if len(normalized) > 120 and len(old_normalized) > 120:

            ratio = SequenceMatcher(
                None,
                normalized[:3000],
                old_normalized[:3000]
            ).ratio()

            if ratio >= 0.90:
                return True

    return False


# ============================================================
# PROCESS ONE COLUMN
# ============================================================

def process_column(column_image):

    items = ocr_column(
        column_image
    )

    if not items:
        return []

    lines = build_lines(
        items
    )

    if not lines:
        return []

    articles = split_column_into_articles(
        lines,
        column_image.height
    )

    results = []

    for article_lines in articles:

        text = article_to_text(
            article_lines
        )

        if not article_quality(text):
            continue

        analysis = defence_analysis(
            text
        )

        if not analysis["is_defence"]:
            continue

        results.append(
            {
                "text": text,
                "analysis": analysis,
                "lines": article_lines
            }
        )

    return results


# ============================================================
# PROCESS PAGE
# ============================================================

def process_page(page, page_number):

    original = render_pdf_page(
        page,
        zoom=2.4
    )

    image = preprocess_image(
        original
    )

    columns = detect_columns(
        image
    )

    page_results = []

    for column_index, bounds in enumerate(columns, start=1):

        column_image = crop_column(
            image,
            bounds
        )

        articles = process_column(
            column_image
        )

        for article in articles:

            article["page"] = page_number
            article["column"] = column_index

            page_results.append(
                article
            )

    return page_results


# ============================================================
# EXTRA SAFETY FILTER
# ============================================================

def final_defence_check(article):

    text = article["text"]

    analysis = defence_analysis(
        text
    )

    if not analysis["is_defence"]:
        return False

    # If article has clear non-defence topic and
    # only weak military vocabulary, reject.
    low = normalise_for_matching(text)

    strong_hits = count_phrase_hits(
        text,
        VERY_STRONG_PHRASES
    )

    military_hits = count_term_hits(
        text,
        STRONG_TERMS
    )

    non_defence_hits = count_term_hits(
        text,
        NON_DEFENCE_TOPICS
    )

    if (
        len(strong_hits) == 0
        and len(military_hits) <= 3
        and len(non_defence_hits) >= 3
    ):
        return False

    # Context-only article should never pass.
    if (
        not strong_hits
        and len(military_hits) <= 1
    ):
        return False

    return True


# ============================================================
# DISPLAY TEXT
# ============================================================

def format_article_text(text):

    text = clean_text(text)

    # Try to restore paragraph-like breaks after sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text
    )

    if len(sentences) >= 3:

        paragraphs = []

        current = []

        for sentence in sentences:

            current.append(sentence)

            if len(current) >= 3:

                paragraphs.append(
                    " ".join(current)
                )

                current = []

        if current:
            paragraphs.append(
                " ".join(current)
            )

        return "\n\n".join(
            paragraphs
        )

    return text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Scanner Settings")

    st.markdown(
        """
**OCR engine:** Tesseract

**Layout mode:** Column-aware

**Filtering:** Strict article-level

**Output:** Complete OCR article

**Country names alone:** Not enough

**Single military word:** Not enough
"""
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload Newspaper PDF",
    type=["pdf"],
    help="Upload an English newspaper PDF."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded:

    pdf_bytes = uploaded.read()

    try:

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

    except Exception as e:

        st.error(
            f"Could not open PDF: {e}"
        )

        st.stop()

    st.success(
        f"📄 {uploaded.name} | {len(document)} pages"
    )

    st.subheader(
        "Select scan range"
    )

    scan_mode = st.radio(
        "Pages to scan",
        [
            "First 3 pages - TEST",
            "All pages",
            "Select pages"
        ],
        horizontal=True
    )

    if scan_mode == "First 3 pages - TEST":

        selected_pages = list(
            range(
                min(3, len(document))
            )
        )

    elif scan_mode == "All pages":

        selected_pages = list(
            range(len(document))
        )

    else:

        page_numbers = st.multiselect(
            "Select page numbers",
            options=list(
                range(
                    1,
                    len(document) + 1
                )
            ),
            default=[1]
        )

        selected_pages = [
            p - 1
            for p in page_numbers
        ]

    st.write(
        f"**Pages selected:** {len(selected_pages)}"
    )

    scan_button = st.button(
        "🔎 Scan Defence News",
        type="primary",
        use_container_width=True
    )

    if scan_button:

        if not selected_pages:

            st.warning(
                "Please select at least one page."
            )

            st.stop()

        all_results = []

        progress = st.progress(
            0
        )

        status = st.empty()

        for position, page_index in enumerate(
            selected_pages,
            start=1
        ):

            status.text(
                f"🔍 Scanning page "
                f"{page_index + 1} "
                f"({position}/{len(selected_pages)})..."
            )

            try:

                page = document.load_page(
                    page_index
                )

                page_results = process_page(
                    page,
                    page_index + 1
                )

                all_results.extend(
                    page_results
                )

            except Exception as e:

                st.warning(
                    f"Page {page_index + 1} "
                    f"could not be processed: {e}"
                )

            progress.progress(
                position / len(selected_pages)
            )

        status.text(
            "✅ Scan completed."
        )

        # ====================================================
        # FINAL DEDUPLICATION
        # ====================================================

        final_results = []

        previous_texts = []

        for result in all_results:

            text = result["text"]

            if not final_defence_check(
                result
            ):
                continue

            if is_duplicate(
                text,
                previous_texts
            ):
                continue

            previous_texts.append(
                text
            )

            final_results.append(
                result
            )

        st.session_state["defence_results"] = final_results


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "defence_results" in st.session_state:

    results = st.session_state[
        "defence_results"
    ]

    st.success(
        f"Scan completed — "
        f"{len(results)} defence articles found."
    )

    st.divider()

    st.header(
        "📰 Defence News Found"
    )

    st.caption(
        "Only articles passing the strict defence relevance filter are shown."
    )

    if results:

        for index, result in enumerate(
            results,
            start=1
        ):

            analysis = result["analysis"]

            st.subheader(
                f"News {index}"
            )

            st.caption(
                f"📄 Page {result['page']} "
                f"• Column {result['column']}"
            )

            signals = analysis.get(
                "signals",
                []
            )

            # Only show useful signals.
            display_signals = []

            for signal in signals:

                if signal not in display_signals:

                    display_signals.append(
                        signal
                    )

            if display_signals:

                st.write(
                    "🔎 **Defence signals:** "
                    + ", ".join(
                        display_signals[:8]
                    )
                )

            # Full article.
            article_display = format_article_text(
                result["text"]
            )

            st.markdown(
                article_display
            )

            st.divider()

        # ====================================================
        # DOWNLOAD
        # ====================================================

        download_parts = []

        download_parts.append(
            "DEFENCE NEWS SCANNER OCR\n"
        )

        download_parts.append(
            "=" * 70
        )

        download_parts.append(
            "\n\n"
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            download_parts.append(
                f"NEWS {index}\n"
            )

            download_parts.append(
                f"Page: {result['page']} | "
                f"Column: {result['column']}\n"
            )

            download_parts.append(
                "-" * 70
            )

            download_parts.append(
                "\n"
            )

            download_parts.append(
                result["text"]
            )

            download_parts.append(
                "\n\n"
            )

        download_text = "".join(
            download_parts
        )

        st.download_button(
            label="⬇️ Download Extracted Defence News",
            data=download_text,
            file_name="defence_news_extracted.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:

        st.info(
            "No defence-related article was detected "
            "in the selected pages."
        )

else:

    if uploaded:

        st.info(
            "Select the pages and click "
            "'🔎 Scan Defence News'."
        )

    else:

        st.info(
            "Upload a newspaper PDF to begin."
        )
