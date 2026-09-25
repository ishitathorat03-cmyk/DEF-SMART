import io
import re
import hashlib
from collections import defaultdict

import fitz  # PyMuPDF
import pytesseract
import streamlit as st

from PIL import Image, ImageOps, ImageFilter


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

OCR_CONFIG = "--oem 3 --psm 3"

# Keep this low enough to catch short genuine defence articles.
MIN_ARTICLE_WORDS = 25

MAX_ARTICLES_PER_PAGE = 20


# ============================================================
# DEFENCE VOCABULARY
# ============================================================

# These are phrases that strongly indicate that the article
# itself is about defence / military affairs.

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
    "fighter jets",
    "fighter jet",
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
]


# Supporting defence terms.
STRONG_TERMS = [
    "army",
    "navy",
    "military",
    "missile",
    "missiles",
    "drone",
    "drones",
    "troops",
    "troop",
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
    "navy",
    "military",
    "security forces",
]


# Topics which frequently contain defence-related words
# but are NOT automatically defence articles.

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
    "election",
    "elections",
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
# BASIC TEXT FUNCTIONS
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_line(text):
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)

    return text.strip()


def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_phrase(text, phrase):
    return len(
        re.findall(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
            text
        )
    )


# ============================================================
# DEFENCE ARTICLE CLASSIFIER
# ============================================================

def defence_analysis(text):
    """
    Strict article-level defence classifier.

    Important:
    A country name, war word, or one military word alone
    should NOT make an article defence-related.
    """

    text_norm = normalize_text(text).lower()

    if not text_norm:
        return False, [], 0

    words = word_count(text_norm)

    if words < MIN_ARTICLE_WORDS:
        return False, [], 0

    # --------------------------------------------------------
    # Count very strong phrases
    # --------------------------------------------------------

    very_strong_hits = []

    for phrase in VERY_STRONG_PHRASES:
        if phrase in text_norm:
            very_strong_hits.append(phrase)

    # --------------------------------------------------------
    # Count supporting defence terms
    # --------------------------------------------------------

    strong_hits = []

    for term in STRONG_TERMS:
        if re.search(
            r"(?<!\w)" + re.escape(term) + r"(?!\w)",
            text_norm
        ):
            strong_hits.append(term)

    # Remove duplicates
    strong_hits = list(dict.fromkeys(strong_hits))
    very_strong_hits = list(dict.fromkeys(very_strong_hits))

    # --------------------------------------------------------
    # Count non-defence context
    # --------------------------------------------------------

    non_defence_hits = []

    for topic in NON_DEFENCE_TOPICS:
        if topic in text_norm:
            non_defence_hits.append(topic)

    non_defence_hits = list(dict.fromkeys(non_defence_hits))

    # --------------------------------------------------------
    # Defence density
    # --------------------------------------------------------

    total_defence_hits = len(very_strong_hits) + len(strong_hits)

    density = total_defence_hits / max(words, 1)

    # --------------------------------------------------------
    # RULE 1
    # Very weak evidence = reject
    # --------------------------------------------------------

    if len(very_strong_hits) == 0 and len(strong_hits) < 4:
        return False, [], 0

    # --------------------------------------------------------
    # RULE 2
    # Generic diplomacy / interview / profile articles
    # should not pass just because they contain LAC/war/etc.
    # --------------------------------------------------------

    political_context = any(
        x in text_norm
        for x in [
            "diplomatic",
            "diplomacy",
            "foreign policy",
            "foreign relations",
            "bilateral",
            "interview",
            "newsmakers",
            "political",
            "politics",
            "visa",
            "economy",
        ]
    )

    if political_context:
        if len(very_strong_hits) < 2 and len(strong_hits) < 4:
            return False, [], 0

        if density < 0.025:
            return False, [], 0

    # --------------------------------------------------------
    # RULE 3
    # Culture / entertainment
    # --------------------------------------------------------

    culture_context = any(
        x in text_norm
        for x in [
            "dance",
            "bharatanatyam",
            "music",
            "film",
            "movie",
            "actor",
            "actress",
            "singer",
            "culture",
            "cultural",
        ]
    )

    if culture_context:
        if len(very_strong_hits) < 2 and len(strong_hits) < 4:
            return False, [], 0

        if density < 0.03:
            return False, [], 0

    # --------------------------------------------------------
    # RULE 4
    # Rescue / disaster stories
    # --------------------------------------------------------

    disaster_context = any(
        x in text_norm
        for x in [
            "rescue",
            "rescued",
            "hostel",
            "hospital",
            "volcano",
            "earthquake",
            "flood",
            "collapse",
        ]
    )

    if disaster_context:
        if len(very_strong_hits) < 2 and len(strong_hits) < 4:
            return False, [], 0

        if density < 0.03:
            return False, [], 0

    # --------------------------------------------------------
    # RULE 5
    # Strong acceptance rules
    # --------------------------------------------------------

    accepted = False

    # Multiple explicit military phrases.
    if len(very_strong_hits) >= 2:
        accepted = True

    # One explicit phrase + several military terms.
    elif len(very_strong_hits) >= 1 and len(strong_hits) >= 4:
        accepted = True

    # Several military terms with reasonable density.
    elif len(strong_hits) >= 5 and density >= 0.025:
        accepted = True

    if not accepted:
        return False, [], 0

    # --------------------------------------------------------
    # Signals shown in UI
    # --------------------------------------------------------

    signals = []

    signals.extend(very_strong_hits)
    signals.extend(strong_hits)

    signals = list(dict.fromkeys(signals))

    return True, signals[:10], len(signals)


# ============================================================
# PDF NATIVE TEXT EXTRACTION
# ============================================================

def extract_native_blocks(page):
    """
    Extract text blocks directly from PDF.

    This is much cleaner than OCR when the PDF contains
    selectable text.
    """

    blocks = []

    try:
        data = page.get_text("dict")
    except Exception:
        return []

    for block in data.get("blocks", []):

        if block.get("type") != 0:
            continue

        bbox = block.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        lines = []

        font_sizes = []

        for line in block.get("lines", []):

            line_parts = []

            for span in line.get("spans", []):

                txt = span.get("text", "")

                if txt.strip():
                    line_parts.append(txt)

                size = span.get("size")

                if size:
                    font_sizes.append(float(size))

            if line_parts:
                line_text = clean_line(" ".join(line_parts))

                if line_text:
                    lines.append(line_text)

        if not lines:
            continue

        text = "\n".join(lines).strip()

        if not text:
            continue

        avg_size = (
            sum(font_sizes) / len(font_sizes)
            if font_sizes
            else 10
        )

        blocks.append({
            "x0": float(bbox[0]),
            "y0": float(bbox[1]),
            "x1": float(bbox[2]),
            "y1": float(bbox[3]),
            "width": float(bbox[2] - bbox[0]),
            "height": float(bbox[3] - bbox[1]),
            "text": text,
            "avg_size": avg_size,
            "source": "native",
        })

    return blocks


# ============================================================
# IMAGE RENDERING
# ============================================================

def render_page(page, dpi=220):

    zoom = dpi / 72.0

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.open(
        io.BytesIO(pix.tobytes("png"))
    )

    return image


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_image(image):

    gray = ImageOps.grayscale(image)

    # Slight contrast improvement
    gray = ImageOps.autocontrast(gray)

    # Light sharpening
    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# ============================================================
# OCR BLOCK EXTRACTION
# ============================================================

def ocr_blocks(image):

    processed = preprocess_image(image)

    try:
        data = pytesseract.image_to_data(
            processed,
            config=OCR_CONFIG,
            output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        st.error(
            f"OCR error: {e}"
        )
        return []

    grouped = defaultdict(list)

    n = len(data["text"])

    for i in range(n):

        txt = data["text"][i].strip()

        if not txt:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0

        if conf < 20:
            continue

        block_num = data["block_num"][i]

        grouped[block_num].append({
            "text": txt,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": conf,
        })

    blocks = []

    for _, words in grouped.items():

        if not words:
            continue

        words = sorted(
            words,
            key=lambda x: (
                x["top"],
                x["left"]
            )
        )

        # Group words approximately by line.
        lines = []

        for word in words:

            placed = False

            for line in lines:

                if abs(
                    word["top"] - line["top"]
                ) <= max(
                    12,
                    word["height"] * 0.7
                ):
                    line["words"].append(word)
                    placed = True
                    break

            if not placed:

                lines.append({
                    "top": word["top"],
                    "words": [word]
                })

        text_lines = []

        for line in lines:

            line_words = sorted(
                line["words"],
                key=lambda x: x["left"]
            )

            line_text = " ".join(
                x["text"]
                for x in line_words
            )

            if line_text.strip():
                text_lines.append(
                    line_text.strip()
                )

        if not text_lines:
            continue

        x0 = min(
            x["left"]
            for x in words
        )

        y0 = min(
            x["top"]
            for x in words
        )

        x1 = max(
            x["left"] + x["width"]
            for x in words
        )

        y1 = max(
            x["top"] + x["height"]
            for x in words
        )

        avg_height = sum(
            x["height"]
            for x in words
        ) / len(words)

        avg_conf = sum(
            x["conf"]
            for x in words
        ) / len(words)

        blocks.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "width": x1 - x0,
            "height": y1 - y0,
            "text": "\n".join(text_lines),
            "avg_size": avg_height,
            "confidence": avg_conf,
            "source": "ocr",
        })

    return blocks


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(blocks):

    if not blocks:
        return []

    # Sort from left to right.
    sorted_blocks = sorted(
        blocks,
        key=lambda b: b["x0"]
    )

    page_width = max(
        b["x1"]
        for b in blocks
    )

    # Approximate newspaper column width.
    typical_width = sorted(
        b["width"]
        for b in blocks
    )

    if typical_width:
        median_width = typical_width[
            len(typical_width) // 2
        ]
    else:
        median_width = page_width / 4

    # Maximum expected number of columns.
    max_columns = 5

    # Dynamic x-gap used to split columns.
    gap_threshold = max(
        35,
        median_width * 0.20
    )

    columns = []

    for block in sorted_blocks:

        placed = False

        center = (
            block["x0"] +
            block["x1"]
        ) / 2

        for column in columns:

            c_left = column["left"]
            c_right = column["right"]

            # If block center is reasonably close to this
            # column's center, keep it there.
            c_center = (
                c_left +
                c_right
            ) / 2

            if abs(center - c_center) <= (
                max(
                    median_width * 0.45,
                    50
                )
            ):

                column["blocks"].append(
                    block
                )

                column["left"] = min(
                    column["left"],
                    block["x0"]
                )

                column["right"] = max(
                    column["right"],
                    block["x1"]
                )

                placed = True
                break

        if not placed:

            if len(columns) < max_columns:

                columns.append({
                    "left": block["x0"],
                    "right": block["x1"],
                    "blocks": [block]
                })

    # Sort columns left to right.
    columns.sort(
        key=lambda c: c["left"]
    )

    # Sort blocks top-to-bottom.
    for column in columns:
        column["blocks"].sort(
            key=lambda b: (
                b["y0"],
                b["x0"]
            )
        )

    return columns


# ============================================================
# HEADLINE DETECTION
# ============================================================

def looks_like_headline(block):

    text = clean_line(
        block.get("text", "")
    )

    if not text:
        return False

    lower = text.lower()

    # Remove punctuation when checking section labels.
    label = re.sub(
        r"[^a-z0-9 ]",
        "",
        lower
    ).strip()

    # Section labels must NOT become article headlines.
    if label in SECTION_LABELS:
        return False

    words = text.split()

    if len(words) > 18:
        return False

    avg_size = float(
        block.get("avg_size", 10)
    )

    source = block.get(
        "source",
        "native"
    )

    # Native PDF
    if source == "native":

        if avg_size >= 17:
            return True

        if avg_size >= 14 and len(words) <= 12:
            return True

    # OCR
    else:

        if avg_size >= 24:
            return True

        if avg_size >= 19 and len(words) <= 12:
            return True

    # All caps headline.
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
# BLOCK TEXT
# ============================================================

def block_text(block):

    text = block.get(
        "text",
        ""
    )

    text = clean_line(text)

    return text


# ============================================================
# ARTICLE CANDIDATE BUILDER
# ============================================================

def build_article_candidates(blocks):

    if not blocks:
        return []

    blocks = sorted(
        blocks,
        key=lambda b: (
            b["y0"],
            b["x0"]
        )
    )

    candidates = []

    current = []

    previous = None

    for block in blocks:

        text = block_text(block)

        if not text:
            continue

        is_headline = looks_like_headline(
            block
        )

        # ----------------------------------------------------
        # First block
        # ----------------------------------------------------

        if not current:

            current = [block]
            previous = block
            continue

        previous_height = max(
            previous.get(
                "height",
                10
            ),
            10
        )

        vertical_gap = (
            block["y0"] -
            previous["y1"]
        )

        # ----------------------------------------------------
        # Strong article boundary:
        # new large headline after existing article.
        # ----------------------------------------------------

        new_article_headline = (
            is_headline
            and len(current) >= 2
            and block.get(
                "height",
                10
            ) >= previous_height * 1.25
        )

        # ----------------------------------------------------
        # Large whitespace gap.
        # ----------------------------------------------------

        definite_gap = (
            vertical_gap >
            max(
                45,
                previous_height * 3
            )
        )

        large_gap = (
            vertical_gap >
            max(
                18,
                previous_height * 1.8
            )
            and len(current) >= 3
        )

        if new_article_headline or definite_gap:

            candidates.append(
                current
            )

            current = [block]

        elif large_gap and is_headline:

            candidates.append(
                current
            )

            current = [block]

        else:

            current.append(block)

        previous = block

    if current:
        candidates.append(
            current
        )

    return candidates


# ============================================================
# ARTICLE RECONSTRUCTION
# ============================================================

def reconstruct_article(blocks):

    if not blocks:
        return ""

    blocks = sorted(
        blocks,
        key=lambda b: (
            b["y0"],
            b["x0"]
        )
    )

    parts = []

    for block in blocks:

        text = block_text(block)

        if not text:
            continue

        # Preserve paragraph-like breaks.
        if parts:
            parts.append("")

        parts.append(text)

    text = "\n".join(parts)

    return clean_article(text)


# ============================================================
# ARTICLE CLEANING
# ============================================================

def clean_article(text):

    if not text:
        return ""

    lines = [
        clean_line(x)
        for x in text.splitlines()
    ]

    lines = [
        x for x in lines
        if x
    ]

    # Remove obvious repeated OCR garbage.
    cleaned = []

    previous = ""

    for line in lines:

        if line.lower() == previous.lower():
            continue

        # Skip tiny page-number-only lines.
        if re.fullmatch(
            r"[\d\W]{1,5}",
            line
        ):
            continue

        cleaned.append(line)

        previous = line

    return "\n".join(
        cleaned
    ).strip()


# ============================================================
# ARTICLE QUALITY
# ============================================================

def article_quality(text):

    if not text:
        return 0

    words = word_count(text)

    if words < MIN_ARTICLE_WORDS:
        return 0

    score = 0

    # Reasonable article length.
    if words >= 40:
        score += 2

    if words >= 80:
        score += 2

    if words >= 150:
        score += 1

    # Sentence structure.
    sentences = re.split(
        r"[.!?]+",
        text
    )

    real_sentences = [
        s for s in sentences
        if word_count(s) >= 5
    ]

    if len(real_sentences) >= 2:
        score += 2

    if len(real_sentences) >= 4:
        score += 1

    # Avoid huge single-line garbage.
    line_lengths = [
        len(x)
        for x in text.splitlines()
        if x.strip()
    ]

    if line_lengths:

        avg_line = (
            sum(line_lengths) /
            len(line_lengths)
        )

        if avg_line < 500:
            score += 1

    return score


# ============================================================
# FINGERPRINT
# ============================================================

def fingerprint(text):

    normalized = normalize_text(
        text
    ).lower()

    # Remove punctuation.
    normalized = re.sub(
        r"[^a-z0-9 ]",
        " ",
        normalized
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()

    return hashlib.md5(
        normalized.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(articles):

    unique = []

    fingerprints = set()

    for article in articles:

        fp = fingerprint(
            article["text"]
        )

        if fp in fingerprints:
            continue

        # Basic containment check.
        duplicate = False

        current_words = set(
            normalize_text(
                article["text"]
            ).lower().split()
        )

        if len(current_words) > 20:

            for existing in unique:

                existing_words = set(
                    normalize_text(
                        existing["text"]
                    ).lower().split()
                )

                if not existing_words:
                    continue

                intersection = (
                    len(
                        current_words &
                        existing_words
                    )
                )

                smaller = min(
                    len(current_words),
                    len(existing_words)
                )

                if (
                    smaller > 20
                    and
                    intersection /
                    smaller
                    > 0.88
                ):
                    duplicate = True
                    break

        if duplicate:
            continue

        fingerprints.add(fp)
        unique.append(article)

    return unique


# ============================================================
# PAGE PROCESSING
# ============================================================

def process_page(
    page,
    page_number
):

    native_blocks = extract_native_blocks(
        page
    )

    # Native PDF is preferred if it contains
    # enough readable text.
    native_words = word_count(
        " ".join(
            b["text"]
            for b in native_blocks
        )
    )

    if native_words >= 40:

        blocks = native_blocks
        source = "Native PDF"

    else:

        image = render_page(
            page,
            dpi=220
        )

        blocks = ocr_blocks(
            image
        )

        source = "OCR"

    if not blocks:
        return [], source

    columns = detect_columns(
        blocks
    )

    page_articles = []

    # --------------------------------------------------------
    # Process each newspaper column separately.
    # --------------------------------------------------------

    for column_index, column in enumerate(
        columns,
        start=1
    ):

        column_blocks = column[
            "blocks"
        ]

        candidates = build_article_candidates(
            column_blocks
        )

        for candidate_index, candidate in enumerate(
            candidates,
            start=1
        ):

            text = reconstruct_article(
                candidate
            )

            if not text:
                continue

            words = word_count(
                text
            )

            if words < MIN_ARTICLE_WORDS:
                continue

            quality = article_quality(
                text
            )

            if quality < 2:
                continue

            is_defence, signals, signal_count = (
                defence_analysis(
                    text
                )
            )

            if not is_defence:
                continue

            page_articles.append({
                "page": page_number,
                "column": column_index,
                "candidate": candidate_index,
                "source": source,
                "text": text,
                "signals": signals,
                "quality": quality,
                "words": words,
            })

    return page_articles, source


# ============================================================
# SCAN COMPLETE PDF
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

    native_count = 0
    ocr_count = 0

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
            native_count += 1
        else:
            ocr_count += 1

        if progress_callback:
            progress_callback(
                (index + 1) / total
            )

    document.close()

    # Remove duplicates.
    all_articles = remove_duplicates(
        all_articles
    )

    # Sort by page then column.
    all_articles.sort(
        key=lambda x: (
            x["page"],
            x["column"]
        )
    )

    return (
        all_articles,
        native_count,
        ocr_count
    )


# ============================================================
# TEXT EXPORT
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

    for i, article in enumerate(
        articles,
        start=1
    ):

        output.append(
            f"NEWS {i}"
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
            f"Signals: {', '.join(article['signals'])}"
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
# STREAMLIT UI
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "Upload a newspaper PDF and extract complete defence-related articles."
)

st.markdown(
    """
### How it works

**PDF → Page detection → Column detection → Article separation → OCR/Text extraction → Defence filtering**

The scanner is designed to reject stories that only mention words such as
"army", "war", "operation", or "border" while actually being about unrelated
subjects.
"""
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

    st.markdown(
        "---"
    )

    st.info(
        "Native PDF text is used when available. "
        "OCR is used automatically for scanned/image pages."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload newspaper PDF",
    type=["pdf"]
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    try:

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total_pages = len(doc)

        doc.close()

    except Exception as e:

        st.error(
            f"Could not open PDF: {e}"
        )

        st.stop()

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    st.write(
        f"📄 Pages in PDF: **{total_pages}**"
    )

    # --------------------------------------------------------
    # Page selection
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
    # Scan button
    # --------------------------------------------------------

    scan_button = st.button(
        "🔍 Scan Newspaper",
        type="primary",
        use_container_width=True
    )

    if scan_button:

        if not selected_pages:

            st.warning(
                "Please select at least one page."
            )

            st.stop()

        progress_bar = st.progress(
            0
        )

        status_text = st.empty()

        status_text.info(
            "Starting newspaper scan..."
        )

        def update_progress(value):

            progress_bar.progress(
                min(
                    max(
                        value,
                        0
                    ),
                    1
                )
            )

            status_text.info(
                f"Scanning newspaper... "
                f"{int(value * 100)}%"
            )

        try:

            (
                articles,
                native_count,
                ocr_count
            ) = scan_pdf(
                pdf_bytes,
                selected_pages,
                progress_callback=update_progress
            )

        except Exception as e:

            progress_bar.empty()

            status_text.empty()

            st.error(
                f"Scan failed: {e}"
            )

            st.stop()

        progress_bar.progress(
            1.0
        )

        status_text.success(
            "Newspaper scan completed!"
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

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
        # Results
        # ----------------------------------------------------

        if not articles:

            st.warning(
                "No genuine defence-related articles were detected in the selected pages."
            )

            st.info(
                "Try scanning more pages or check whether the newspaper PDF "
                "contains selectable text / readable scans."
            )

        else:

            st.success(
                f"Found {len(articles)} defence-related article(s)."
            )

            for index, article in enumerate(
                articles,
                start=1
            ):

                signals = ", ".join(
                    article["signals"]
                )

                title_text = (
                    f"NEWS {index}  •  "
                    f"Page {article['page']}  •  "
                    f"Column {article['column']}"
                )

                with st.expander(
                    title_text,
                    expanded=True
                ):

                    # Metadata
                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.write(
                            f"**Source:** {article['source']}"
                        )

                    with m2:
                        st.write(
                            f"**Words:** {article['words']}"
                        )

                    with m3:
                        st.write(
                            f"**Quality:** {article['quality']}"
                        )

                    st.write(
                        f"**Defence signals:** {signals}"
                    )

                    st.markdown(
                        "---"
                    )

                    # Full article text.
                    st.text(
                        article["text"]
                    )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            download_text = make_download_text(
                articles
            )

            st.markdown(
                "---"
            )

            st.download_button(
                label="⬇️ Download Defence Articles",
                data=download_text,
                file_name="defence_news_results.txt",
                mime="text/plain",
                use_container_width=True
            )
