import io
import re
import hashlib
from collections import defaultdict

import fitz
import pytesseract
import streamlit as st

from PIL import Image, ImageOps, ImageFilter


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)

OCR_CONFIG = "--oem 3 --psm 3"

MIN_ARTICLE_WORDS = 18
MIN_BODY_WORDS = 18

# Maximum number of article candidates allowed on one page
MAX_ARTICLES_PER_PAGE = 30


# ============================================================
# DEFENCE VOCABULARY
# ============================================================

# These phrases have strong military meaning.
VERY_STRONG_PHRASES = [
    "indian army",
    "indian navy",
    "indian air force",
    "armed forces",
    "corps commander",
    "corps commanders",
    "military operation",
    "military operations",
    "military exercise",
    "military exercises",
    "military deployment",
    "military deployments",
    "military talks",
    "military forces",
    "military officials",
    "air defence",
    "air defense",
    "missile strike",
    "missile strikes",
    "missile attack",
    "missile attacks",
    "drone strike",
    "drone strikes",
    "drone attack",
    "drone attacks",
    "naval exercise",
    "naval exercises",
    "defence ministry",
    "defense ministry",
    "ministry of defence",
    "ministry of defense",
    "border security",
    "border forces",
    "border troops",
    "troop deployment",
    "troop deployments",
    "troop movement",
    "troop movements",
    "line of actual control",
    "line of control",
    "special forces",
    "fighter aircraft",
    "fighter aircrafts",
    "fighter jet",
    "fighter jets",
    "warship",
    "warships",
    "airbase",
    "air base",
    "submarine",
    "submarines",
    "artillery unit",
    "artillery units",
    "military unit",
    "military units",
    "military commander",
    "military commanders",
    "military personnel",
    "military drills",
    "military drill",
    "army troops",
    "navy ships",
    "air force aircraft",
    "security forces",
    "armed forces personnel",
]

STRONG_TERMS = [
    "army",
    "navy",
    "military",
    "missile",
    "missiles",
    "drone",
    "drones",
    "troop",
    "troops",
    "soldier",
    "soldiers",
    "commander",
    "commanders",
    "brigade",
    "brigades",
    "battalion",
    "battalions",
    "regiment",
    "regiments",
    "artillery",
    "warship",
    "warships",
    "frigate",
    "frigates",
    "submarine",
    "submarines",
    "aircraft",
    "fighter",
    "fighters",
    "helicopter",
    "helicopters",
    "airbase",
    "airbases",
    "defence",
    "defense",
    "border",
    "combat",
    "weapon",
    "weapons",
    "ammunition",
    "paramilitary",
    "special forces",
    "corps",
    "irgc",
    "air force",
    "naval",
    "security forces",
]


# ============================================================
# NON-DEFENCE CONTEXT
# ============================================================

NON_DEFENCE_TOPICS = [
    "newsmakers",
    "interview",
    "profile",
    "politics",
    "political",
    "politician",
    "diplomatic ties",
    "diplomatic relation",
    "diplomatic relations",
    "foreign relations",
    "foreign policy",
    "bilateral ties",
    "bilateral relations",
    "economy",
    "economic",
    "business",
    "education",
    "visa",
    "visas",
    "culture",
    "cultural",
    "dance",
    "bharatanatyam",
    "film",
    "films",
    "movie",
    "movies",
    "music",
    "sports",
    "sport",
    "jobs",
    "employment",
    "hospital",
    "hostel",
    "rescue",
    "rescued",
    "volcano",
    "earthquake",
    "flood",
    "weather",
    "school",
    "college",
    "university",
    "actor",
    "actress",
    "singer",
    "festival",
]


SECTION_LABELS = {
    "newsmakers",
    "opinion",
    "business",
    "world",
    "sports",
    "nation",
    "politics",
    "explained",
    "editorial",
    "lifestyle",
    "culture",
    "technology",
    "education",
    "cities",
    "india",
    "world news",
    "business news",
    "sports news",
    "nation news",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def word_count(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text
        )
    )


def contains_phrase(text, phrase):
    return bool(
        re.search(
            r"(?<!\w)"
            + re.escape(phrase)
            + r"(?!\w)",
            text
        )
    )


# ============================================================
# STRICT DEFENCE CLASSIFIER
# ============================================================

# ============================================================
# DEFENCE CLASSIFIER
# ============================================================

def defence_analysis(text):

    text = normalize_text(text).lower()

    words = word_count(text)

    if words < 18:
        return False, [], 0

    # --------------------------------------------------------
    # VERY STRONG DEFENCE PHRASES
    # --------------------------------------------------------

    very_strong = [
        "indian army",
        "indian navy",
        "indian air force",
        "armed forces",
        "corps commander",
        "corps commanders",
        "military operation",
        "military operations",
        "military exercise",
        "military exercises",
        "military deployment",
        "military talks",
        "air defence",
        "air defense",
        "missile strike",
        "missile strikes",
        "missile attack",
        "missile attacks",
        "drone strike",
        "drone strikes",
        "drone attack",
        "drone attacks",
        "naval exercise",
        "naval exercises",
        "defence ministry",
        "defense ministry",
        "ministry of defence",
        "ministry of defense",
        "border security",
        "border forces",
        "troop deployment",
        "special forces",
        "fighter aircraft",
        "fighter jet",
        "fighter jets",
        "warship",
        "warships",
        "airbase",
        "air base",
        "submarine",
        "submarines",
        "artillery unit",
        "military unit",
        "military commander",
        "military commanders",
        "military personnel",
        "security forces",
        "line of actual control",
        "line of control",
    ]

    # --------------------------------------------------------
    # STRONG DEFENCE TERMS
    # --------------------------------------------------------

    strong = [
        "army",
        "navy",
        "military",
        "missile",
        "missiles",
        "drone",
        "drones",
        "troop",
        "troops",
        "soldier",
        "soldiers",
        "commander",
        "commanders",
        "brigade",
        "battalion",
        "regiment",
        "artillery",
        "warship",
        "frigate",
        "submarine",
        "aircraft",
        "fighter",
        "helicopter",
        "airbase",
        "defence",
        "defense",
        "border",
        "combat",
        "weapon",
        "weapons",
        "ammunition",
        "paramilitary",
        "special forces",
        "corps",
        "irgc",
        "air force",
        "naval",
        "lac",
        "loc",
        "operation",
        "operations",
        "forces",
        "deployment",
        "deploy",
        "war",
        "wars",
        "attack",
        "attacks",
        "strike",
        "strikes",
    ]

    # --------------------------------------------------------
    # OBVIOUS NON-DEFENCE CONTEXT
    # --------------------------------------------------------

    non_defence = [
        "newsmakers",
        "bharatanatyam",
        "dance",
        "dancer",
        "music",
        "singer",
        "film",
        "movie",
        "actor",
        "actress",
        "cricket",
        "football",
        "sports",
        "sport",
        "education",
        "student",
        "students",
        "college",
        "school",
        "jobs",
        "employment",
        "business",
        "economy",
        "economic",
        "hospital",
        "hostel",
        "rescue",
        "rescued",
        "volcano",
        "earthquake",
        "flood",
        "weather",
        "culture",
        "cultural",
    ]

    # --------------------------------------------------------
    # FIND SIGNALS
    # --------------------------------------------------------

    phrase_hits = []

    for phrase in very_strong:
        if contains_phrase(text, phrase):
            phrase_hits.append(phrase)

    term_hits = []

    for term in strong:
        if contains_phrase(text, term):
            term_hits.append(term)

    non_hits = []

    for term in non_defence:
        if contains_phrase(text, term):
            non_hits.append(term)

    phrase_hits = list(dict.fromkeys(phrase_hits))
    term_hits = list(dict.fromkeys(term_hits))
    non_hits = list(dict.fromkeys(non_hits))

    # --------------------------------------------------------
    # LOOK AT BEGINNING OF ARTICLE
    # Defence news normally establishes its subject early.
    # --------------------------------------------------------

    first_part = " ".join(
        text.split()[:80]
    )

    first_hits = 0

    for phrase in very_strong:
        if contains_phrase(first_part, phrase):
            first_hits += 3

    for term in strong:
        if contains_phrase(first_part, term):
            first_hits += 1

    # --------------------------------------------------------
    # COUNT ACTUAL OCCURRENCES
    # --------------------------------------------------------

    defence_occurrences = 0

    for term in strong:
        defence_occurrences += len(
            re.findall(
                r"(?<!\w)"
                + re.escape(term)
                + r"(?!\w)",
                text
            )
        )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0

    score += len(phrase_hits) * 5
    score += len(term_hits)

    score += first_hits

    score -= len(non_hits) * 2

    # --------------------------------------------------------
    # RULE 1
    # One very strong military phrase + enough evidence.
    # --------------------------------------------------------

    if len(phrase_hits) >= 1 and score >= 7:

        signals = (
            phrase_hits
            + term_hits
        )

        return (
            True,
            list(dict.fromkeys(signals))[:12],
            score
        )

    # --------------------------------------------------------
    # RULE 2
    # Several military terms occurring repeatedly.
    # --------------------------------------------------------

    if (
        len(term_hits) >= 3
        and defence_occurrences >= 4
    ):

        signals = (
            phrase_hits
            + term_hits
        )

        return (
            True,
            list(dict.fromkeys(signals))[:12],
            score
        )

    # --------------------------------------------------------
    # RULE 3
    # Defence evidence appears near beginning.
    # --------------------------------------------------------

    if (
        len(term_hits) >= 2
        and first_hits >= 3
        and defence_occurrences >= 3
    ):

        signals = (
            phrase_hits
            + term_hits
        )

        return (
            True,
            list(dict.fromkeys(signals))[:12],
            score
        )

    return False, [], score


# ============================================================
# NATIVE PDF WORD EXTRACTION
# ============================================================

def extract_pdf_words(page):

    words = []

    try:
        raw_words = page.get_text(
            "words"
        )
    except Exception:
        return []

    for item in raw_words:

        if len(item) < 8:
            continue

        x0, y0, x1, y1, text = item[:5]

        text = clean_text(
            str(text)
        )

        if not text:
            continue

        words.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "text": text,
            "height": float(
                y1 - y0
            ),
            "source": "native",
        })

    return words


# ============================================================
# RENDER PAGE FOR OCR
# ============================================================

def render_page(page, dpi=220):

    zoom = dpi / 72

    matrix = fitz.Matrix(
        zoom,
        zoom
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    return Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = ImageOps.grayscale(
        image
    )

    image = ImageOps.autocontrast(
        image
    )

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# ============================================================
# OCR WORD EXTRACTION
# ============================================================

def extract_ocr_words(image):

    image = preprocess_image(
        image
    )

    try:

        data = pytesseract.image_to_data(
            image,
            config=OCR_CONFIG,
            output_type=pytesseract.Output.DICT
        )

    except Exception as e:

        st.error(
            f"Tesseract OCR error: {e}"
        )

        return []

    words = []

    total = len(
        data["text"]
    )

    for i in range(total):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            confidence = float(
                data["conf"][i]
            )
        except Exception:
            confidence = 0

        if confidence < 25:
            continue

        left = int(
            data["left"][i]
        )

        top = int(
            data["top"][i]
        )

        width = int(
            data["width"][i]
        )

        height = int(
            data["height"][i]
        )

        words.append({
            "x0": left,
            "y0": top,
            "x1": left + width,
            "y1": top + height,
            "text": text,
            "height": height,
            "confidence": confidence,
            "source": "ocr",
        })

    return words


# ============================================================
# LINE RECONSTRUCTION
# ============================================================

def build_lines(words):

    if not words:
        return []

    words = sorted(
        words,
        key=lambda w: (
            w["y0"],
            w["x0"]
        )
    )

    lines = []

    for word in words:

        placed = False

        word_center = (
            word["y0"]
            + word["y1"]
        ) / 2

        for line in lines:

            line_center = line[
                "center_y"
            ]

            tolerance = max(
                4,
                word["height"] * 0.65
            )

            if abs(
                word_center - line_center
            ) <= tolerance:

                line["words"].append(
                    word
                )

                # Update line center.
                centers = [
                    (
                        w["y0"]
                        + w["y1"]
                    ) / 2
                    for w in line["words"]
                ]

                line["center_y"] = (
                    sum(centers)
                    / len(centers)
                )

                placed = True
                break

        if not placed:

            lines.append({
                "words": [word],
                "center_y": word_center,
            })

    # Sort words within each line.
    for line in lines:

        line["words"].sort(
            key=lambda w: w["x0"]
        )

        line["x0"] = min(
            w["x0"]
            for w in line["words"]
        )

        line["x1"] = max(
            w["x1"]
            for w in line["words"]
        )

        line["y0"] = min(
            w["y0"]
            for w in line["words"]
        )

        line["y1"] = max(
            w["y1"]
            for w in line["words"]
        )

        line["height"] = (
            line["y1"]
            - line["y0"]
        )

        line["text"] = " ".join(
            w["text"]
            for w in line["words"]
        )

    lines.sort(
        key=lambda x: (
            x["y0"],
            x["x0"]
        )
    )

    return lines


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(lines):

    if not lines:
        return []

    page_left = min(
        line["x0"]
        for line in lines
    )

    page_right = max(
        line["x1"]
        for line in lines
    )

    page_width = (
        page_right
        - page_left
    )

    if page_width <= 0:
        return [lines]

    # --------------------------------------------------------
    # Estimate column centers from line centers.
    # --------------------------------------------------------

    centers = []

    for line in lines:

        center = (
            line["x0"]
            + line["x1"]
        ) / 2

        centers.append(
            center
        )

    centers.sort()

    # Find large horizontal gaps.
    gaps = []

    for i in range(
        1,
        len(centers)
    ):

        gap = (
            centers[i]
            - centers[i - 1]
        )

        gaps.append(
            (
                gap,
                centers[i - 1],
                centers[i]
            )
        )

    # Newspaper column gaps are usually
    # substantially larger than ordinary
    # word/line spacing.
    threshold = max(
        45,
        page_width * 0.045
    )

    split_points = []

    for gap, left, right in gaps:

        if gap > threshold:

            split_points.append(
                (left + right) / 2
            )

    # Limit to sensible newspaper columns.
    if len(split_points) > 4:

        split_points = sorted(
            split_points,
            key=lambda x: x
        )[:4]

    boundaries = [
        page_left
    ]

    boundaries.extend(
        split_points
    )

    boundaries.append(
        page_right
    )

    boundaries = sorted(
        boundaries
    )

    columns = []

    for i in range(
        len(boundaries) - 1
    ):

        left = boundaries[i]
        right = boundaries[i + 1]

        column_lines = []

        for line in lines:

            center = (
                line["x0"]
                + line["x1"]
            ) / 2

            if (
                center >= left
                and center <= right
            ):
                column_lines.append(
                    line
                )

        if column_lines:

            columns.append(
                column_lines
            )

    # If detection produced nonsense,
    # fall back to one column.
    if not columns:

        return [lines]

    # Sort each column top-to-bottom.
    for column in columns:

        column.sort(
            key=lambda x: (
                x["y0"],
                x["x0"]
            )
        )

    return columns


# ============================================================
# HEADLINE DETECTION
# ============================================================

def is_section_label(text):

    cleaned = re.sub(
        r"[^a-z0-9 ]",
        "",
        text.lower()
    ).strip()

    return cleaned in SECTION_LABELS


def is_headline(line):

    text = clean_text(
        line["text"]
    )

    if not text:
        return False

    if is_section_label(text):
        return False

    words = text.split()

    if len(words) > 18:
        return False

    height = line.get(
        "height",
        10
    )

    # --------------------------------------------------------
    # Native PDF
    # --------------------------------------------------------

    if line["words"][0].get(
        "source"
    ) == "native":

        if height >= 18:
            return True

        if (
            height >= 14
            and len(words) <= 12
        ):
            return True

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    else:

        if height >= 25:
            return True

        if (
            height >= 19
            and len(words) <= 12
        ):
            return True

    # --------------------------------------------------------
    # ALL CAPS
    # --------------------------------------------------------

    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if (
        len(letters) >= 14
        and len(words) >= 3
        and letters.isupper()
    ):
        return True

    return False


# ============================================================
# ARTICLE SEGMENTATION
# ============================================================

def split_column_into_articles(lines):

    if not lines:
        return []

    lines = sorted(
        lines,
        key=lambda x: (
            x["y0"],
            x["x0"]
        )
    )

    articles = []

    current = []

    for line in lines:

        text = clean_text(
            line["text"]
        )

        if not text:
            continue

        # ----------------------------------------------------
        # First line
        # ----------------------------------------------------

        if not current:

            current.append(
                line
            )

            continue

        previous = current[-1]

        vertical_gap = (
            line["y0"]
            - previous["y1"]
        )

        current_words = word_count(
            " ".join(
                x["text"]
                for x in current
            )
        )

        # ----------------------------------------------------
        # Headline detection
        # ----------------------------------------------------

        headline = is_headline(
            line
        )

        previous_height = max(
            previous["height"],
            1
        )

        # ----------------------------------------------------
        # VERY large whitespace.
        #
        # Only use this when the current article already
        # contains enough text.
        # ----------------------------------------------------

        very_large_gap = (
            vertical_gap
            > max(
                55,
                previous_height * 4.5
            )
        )

        # ----------------------------------------------------
        # Strong headline boundary.
        #
        # Require a significant font-size difference.
        # ----------------------------------------------------

        new_headline = (
            headline
            and current_words >= 60
            and line["height"]
            >= previous["height"] * 1.35
        )

        # ----------------------------------------------------
        # Start a new article.
        # ----------------------------------------------------

        if (
            very_large_gap
            or new_headline
        ):

            articles.append(
                current
            )

            current = [
                line
            ]

        else:

            current.append(
                line
            )

    # --------------------------------------------------------
    # Final article
    # --------------------------------------------------------

    if current:

        articles.append(
            current
        )

    # --------------------------------------------------------
    # REMOVE TINY FRAGMENTS
    # --------------------------------------------------------

    cleaned_articles = []

    for article in articles:

        text = article_text(
            article
        )

        if word_count(text) >= 18:

            cleaned_articles.append(
                article
            )

    return cleaned_articles


# ============================================================
# ARTICLE TEXT
# ============================================================

def article_text(lines):

    if not lines:
        return ""

    output = []

    for line in lines:

        text = clean_text(
            line["text"]
        )

        if text:
            output.append(
                text
            )

    return "\n".join(
        output
    ).strip()


# ============================================================
# ARTICLE QUALITY
# ============================================================

def quality_score(text):

    words = word_count(text)

    if words < MIN_ARTICLE_WORDS:
        return 0

    score = 0

    if words >= 30:
        score += 1

    if words >= 60:
        score += 1

    if words >= 120:
        score += 1

    # Sentence structure.
    sentences = re.split(
        r"[.!?]+",
        text
    )

    valid_sentences = [
        s
        for s in sentences
        if word_count(s) >= 5
    ]

    if len(valid_sentences) >= 2:
        score += 1

    if len(valid_sentences) >= 4:
        score += 1

    return score


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def fingerprint(text):

    text = normalize_text(
        text
    ).lower()

    text = re.sub(
        r"[^a-z0-9 ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return hashlib.md5(
        text.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


def remove_duplicates(articles):

    unique = []

    fingerprints = set()

    for article in articles:

        fp = fingerprint(
            article["text"]
        )

        if fp in fingerprints:
            continue

        # Check if one article is almost
        # completely contained in another.
        duplicate = False

        current_words = set(
            normalize_text(
                article["text"]
            ).lower().split()
        )

        if len(current_words) >= 30:

            for existing in unique:

                existing_words = set(
                    normalize_text(
                        existing["text"]
                    ).lower().split()
                )

                if len(existing_words) < 30:
                    continue

                common = len(
                    current_words
                    & existing_words
                )

                smaller = min(
                    len(current_words),
                    len(existing_words)
                )

                if (
                    common / smaller
                    > 0.88
                ):

                    duplicate = True
                    break

        if duplicate:
            continue

        fingerprints.add(fp)

        unique.append(
            article
        )

    return unique


# ============================================================
# BUILD ARTICLE CANDIDATES FROM WORDS
# ============================================================

def extract_articles_from_words(
    words,
    page_number,
    source
):

    if not words:
        return []

    # --------------------------------------------------------
    # Reconstruct lines
    # --------------------------------------------------------

    lines = build_lines(words)

    if not lines:
        return []

    # --------------------------------------------------------
    # Detect newspaper columns
    # --------------------------------------------------------

    columns = detect_columns(lines)

    if not columns:
        columns = [lines]

    page_articles = []

    # --------------------------------------------------------
    # Process every column
    # --------------------------------------------------------

    for column_number, column in enumerate(
        columns,
        start=1
    ):

        article_candidates = (
            split_column_into_articles(
                column
            )
        )

        # ----------------------------------------------------
        # FALLBACK:
        # If segmentation is too aggressive, create larger
        # candidate windows from the column.
        # ----------------------------------------------------

        if not article_candidates:
            article_candidates = [
                column
            ]

        for candidate_number, candidate in enumerate(
            article_candidates,
            start=1
        ):

            if len(page_articles) >= MAX_ARTICLES_PER_PAGE:
                break

            text = article_text(
                candidate
            )

            words_in_article = word_count(
                text
            )

            if words_in_article < MIN_ARTICLE_WORDS:
                continue

            # ------------------------------------------------
            # Defence classification FIRST
            # ------------------------------------------------
            # Do not reject a candidate only because of the
            # quality score. Newspaper OCR/layout can make
            # sentence reconstruction imperfect.
            # ------------------------------------------------

            is_defence, signals, signal_count = (
                defence_analysis(
                    text
                )
            )

            if not is_defence:
                continue

            quality = quality_score(
                text
            )

            # Accept defence article even when OCR/layout
            # causes a lower quality score.
            if quality < 1:
                quality = 1

            page_articles.append({
                "page": page_number,
                "column": column_number,
                "candidate": candidate_number,
                "source": source,
                "text": text,
                "signals": signals,
                "signal_count": signal_count,
                "quality": quality,
                "words": words_in_article,
            })

    return page_articles


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    page,
    page_number
):

    # ========================================================
    # PASS 1 — NATIVE PDF
    # ========================================================

    native_words = extract_pdf_words(
        page
    )

    native_word_count = len(
        native_words
    )

    # --------------------------------------------------------
    # If native PDF contains enough text, try it first.
    # --------------------------------------------------------

    if native_word_count >= 50:

        native_articles = (
            extract_articles_from_words(
                native_words,
                page_number,
                "Native PDF"
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # If native extraction found defence articles,
        # return immediately.
        # ----------------------------------------------------

        if native_articles:

            return (
                native_articles,
                "Native PDF"
            )

        # ----------------------------------------------------
        # NATIVE EXTRACTION FAILED TO FIND ANY DEFENCE NEWS.
        #
        # This is where the old app stopped.
        #
        # NEW BEHAVIOUR:
        # Render the page and run real OCR.
        # ----------------------------------------------------

        image = render_page(
            page,
            dpi=220
        )

        ocr_words = extract_ocr_words(
            image
        )

        ocr_articles = (
            extract_articles_from_words(
                ocr_words,
                page_number,
                "OCR"
            )
        )

        if ocr_articles:

            return (
                ocr_articles,
                "OCR"
            )

        # Nothing found even after OCR.
        return [], "Native PDF"

    # ========================================================
    # PASS 2 — OCR
    # ========================================================

    image = render_page(
        page,
        dpi=220
    )

    ocr_words = extract_ocr_words(
        image
    )

    ocr_articles = (
        extract_articles_from_words(
            ocr_words,
            page_number,
            "OCR"
        )
    )

    return (
        ocr_articles,
        "OCR"
    )


# ============================================================
# SCAN PDF
# ============================================================

def scan_pdf(
    pdf_bytes,
    selected_pages,
    progress_callback=None
):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    all_articles = []

    native_pages = 0
    ocr_pages = 0

    total = len(
        selected_pages
    )

    for index, page_number in enumerate(
        selected_pages
    ):

        page = document[
            page_number - 1
        ]

        articles, source = process_page(
            page,
            page_number
        )

        all_articles.extend(
            articles
        )

        if source == "Native PDF":
            native_pages += 1
        else:
            ocr_pages += 1

        if progress_callback:

            progress_callback(
                (index + 1) / total
            )

    document.close()

    all_articles = remove_duplicates(
        all_articles
    )

    all_articles.sort(
        key=lambda article: (
            article["page"],
            article["column"],
            article["candidate"]
        )
    )

    return (
        all_articles,
        native_pages,
        ocr_pages
    )


# ============================================================
# DOWNLOAD FILE
# ============================================================

def make_download_text(
    articles
):

    output = []

    output.append(
        "DEFENCE NEWS SCANNER OCR"
    )

    output.append(
        "=" * 70
    )

    output.append("")

    output.append(
        f"Defence articles found: {len(articles)}"
    )

    output.append("")

    for number, article in enumerate(
        articles,
        start=1
    ):

        output.append(
            f"NEWS {number}"
        )

        output.append(
            "-" * 70
        )

        output.append(
            f"Page: {article['page']}"
        )

        output.append(
            f"Column: {article['column']}"
        )

        output.append(
            f"Source: {article['source']}"
        )

        output.append(
            "Defence signals: "
            + ", ".join(
                article["signals"]
            )
        )

        output.append("")

        output.append(
            article["text"]
        )

        output.append("")

        output.append(
            "=" * 70
        )

        output.append("")

    return "\n".join(
        output
    )


# ============================================================
# UI
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.write(
    "Upload a newspaper PDF and extract complete defence-related articles."
)

st.markdown(
    """
**PDF → Word coordinates → Line reconstruction → Column detection → 
Article segmentation → Defence verification → Complete article**
"""
)

st.info(
    "The scanner evaluates the article as a whole. "
    "A single word such as 'war', 'army', 'operation' or 'border' "
    "is not enough to classify an article as defence news."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Scanner Settings"
    )

    scan_mode = st.radio(
        "Pages to scan",
        [
            "First 3 pages TEST",
            "All pages",
            "Select pages"
        ]
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload newspaper PDF",
    type=["pdf"]
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    try:

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total_pages = len(
            document
        )

        document.close()

    except Exception as e:

        st.error(
            f"Unable to open PDF: {e}"
        )

        st.stop()

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    st.write(
        f"📄 Pages in PDF: **{total_pages}**"
    )

    # --------------------------------------------------------
    # PAGE SELECTION
    # --------------------------------------------------------

    if scan_mode == "First 3 pages TEST":

        selected_pages = list(
            range(
                1,
                min(
                    3,
                    total_pages
                ) + 1
            )
        )

    elif scan_mode == "All pages":

        selected_pages = list(
            range(
                1,
                total_pages + 1
            )
        )

    else:

        selected_pages = st.multiselect(
            "Select pages",
            options=list(
                range(
                    1,
                    total_pages + 1
                )
            ),
            default=list(
                range(
                    1,
                    min(
                        3,
                        total_pages
                    ) + 1
                )
            )
        )

    st.write(
        f"**Pages selected:** {len(selected_pages)}"
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    if st.button(
        "🔍 Scan Newspaper",
        type="primary",
        use_container_width=True
    ):

        if not selected_pages:

            st.warning(
                "Select at least one page."
            )

            st.stop()

        progress = st.progress(
            0
        )

        status = st.empty()

        status.info(
            "Reading newspaper layout..."
        )

        def update_progress(value):

            progress = st.progress(
    0
)

status = st.empty()

status.info(
    "Reading newspaper layout..."
)

def update_progress(value):

    progress.progress(
        min(
            max(
                value,
                0
            ),
            1
        )
    )

    status.info(
        f"Scanning newspaper: "
        f"{int(value * len(selected_pages))}"
        f"/{len(selected_pages)} pages..."
    )

try:

    (
        articles,
        native_count,
        ocr_count
    ) = scan_pdf(
        pdf_bytes,
        selected_pages,
        update_progress
    )
        except Exception as e:

            progress.empty()
            status.empty()

            st.exception(e)

            st.stop()

        progress.progress(
            1.0
        )

        status.success(
            "Newspaper scan completed!"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(
            4
        )

        with col1:

            st.metric(
                "Pages scanned",
                len(selected_pages)
            )

        with col2:

            st.metric(
                "Native PDF",
                native_count
            )

        with col3:

            st.metric(
                "OCR",
                ocr_count
            )

        with col4:

            st.metric(
                "Defence articles",
                len(articles)
            )

        st.markdown(
            "---"
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if not articles:

            st.warning(
                "No genuine defence-related articles were detected."
            )

            st.caption(
                "This means the current extraction/classification "
                "pipeline did not find an article satisfying the "
                "strict defence criteria."
            )

        else:

            st.success(
                f"{len(articles)} defence article(s) found."
            )

            for number, article in enumerate(
                articles,
                start=1
            ):

                heading = (
                    f"NEWS {number} • "
                    f"Page {article['page']} • "
                    f"Column {article['column']}"
                )

                with st.expander(
                    heading,
                    expanded=True
                ):

                    c1, c2, c3 = st.columns(
                        3
                    )

                    with c1:

                        st.write(
                            f"**Source:** "
                            f"{article['source']}"
                        )

                    with c2:

                        st.write(
                            f"**Words:** "
                            f"{article['words']}"
                        )

                    with c3:

                        st.write(
                            f"**Quality:** "
                            f"{article['quality']}"
                        )

                    st.write(
                        "**Defence signals:** "
                        + ", ".join(
                            article["signals"]
                        )
                    )

                    st.markdown(
                        "---"
                    )

                    st.text(
                        article["text"]
                    )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            download_data = make_download_text(
                articles
            )

            st.download_button(
                "⬇️ Download Defence Articles",
                data=download_data,
                file_name="defence_news_results.txt",
                mime="text/plain",
                use_container_width=True
            )
