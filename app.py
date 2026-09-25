import io
import re
import hashlib
from collections import defaultdict

import fitz  # PyMuPDF
import pytesseract
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


# ============================================================
# PAGE / OCR SETTINGS
# ============================================================

OCR_CONFIG = "--oem 3 --psm 3"

MIN_ARTICLE_WORDS = 45
MAX_ARTICLES_PER_PAGE = 20

# ============================================================
# DEFENCE LANGUAGE
# ============================================================

VERY_STRONG_PHRASES = [
    "corps commander",
    "corps commanders",
    "military commander",
    "military commanders",
    "armed forces",
    "indian army",
    "indian navy",
    "indian air force",
    "air force",
    "naval forces",
    "navy ship",
    "warship",
    "warships",
    "fighter aircraft",
    "fighter jet",
    "fighter jets",
    "air defence",
    "air defense",
    "missile strike",
    "missile strikes",
    "ballistic missile",
    "ballistic missiles",
    "cruise missile",
    "cruise missiles",
    "drone strike",
    "drone strikes",
    "military operation",
    "military operations",
    "military exercise",
    "military exercises",
    "defence ministry",
    "defense ministry",
    "ministry of defence",
    "ministry of defense",
    "border troops",
    "border forces",
    "border security",
    "line of actual control",
    "lac",
    "loc",
    "line of control",
    "armed conflict",
    "military conflict",
    "military talks",
    "military dialogue",
    "defence cooperation",
    "defense cooperation",
    "military cooperation",
    "special forces",
    "parachute regiment",
    "army regiment",
    "naval exercise",
    "air exercise",
    "joint exercise",
    "joint military exercise",
    "military deployment",
    "troop deployment",
    "troops deployed",
    "military aircraft",
    "naval vessel",
    "naval vessels",
    "air strike",
    "air strikes",
    "airstrike",
    "airstrikes",
]

STRONG_TERMS = [
    "army",
    "navy",
    "military",
    "missile",
    "missiles",
    "drone",
    "drones",
    "troops",
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
    "airbase",
    "defence",
    "defense",
    "military",
    "border",
    "ceasefire",
    "combat",
    "combatant",
    "war",
    "weapons",
    "weapon",
    "ammunition",
    "forces",
    "security forces",
    "paramilitary",
    "special forces",
    "operation",
    "operations",
    "corps",
    "lca",
    "rafale",
    "sukhoi",
    "tejas",
    "ins ",
    "irgc",
]

COUNTRY_CONTEXT = [
    "india",
    "china",
    "pakistan",
    "iran",
    "israel",
    "russia",
    "ukraine",
    "us",
    "united states",
    "north korea",
    "south korea",
]

NON_DEFENCE_TOPICS = [
    "football",
    "cricket",
    "tennis",
    "cinema",
    "film",
    "movie",
    "music",
    "dance",
    "bharatanatyam",
    "theatre",
    "restaurant",
    "food",
    "fashion",
    "education",
    "college admissions",
    "university",
    "jobs",
    "employment",
    "stock market",
    "share market",
    "business",
    "economy",
    "inflation",
    "weather",
    "volcano",
    "earthquake",
    "flood",
    "rescue",
    "hospital",
    "hostel",
    "school",
    "election",
    "elections",
    "polling",
    "political party",
    "actor",
    "actress",
    "singer",
    "artist",
    "book",
    "author",
    "culture",
    "cultural",
    "festival",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")

    # Fix common OCR spacing problems
    text = re.sub(r"(?<=\w)[|¦](?=\w)", "l", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Join words broken by line hyphenation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_line(text):
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text):
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text))


# ============================================================
# DEFENCE CLASSIFIER
# ============================================================

def defence_analysis(text):
    """
    IMPORTANT:
    Classification is performed on the COMPLETE candidate article,
    not on individual OCR lines.
    """

    t = normalize_text(text).lower()

    if word_count(t) < MIN_ARTICLE_WORDS:
        return False, [], 0

    signals = []

    # Very strong phrases
    for phrase in VERY_STRONG_PHRASES:
        if phrase in t:
            signals.append(phrase)

    # Strong individual terms
    strong_found = []

    for term in STRONG_TERMS:
        if term in t:
            strong_found.append(term)

    strong_found = list(dict.fromkeys(strong_found))

    # Country by itself is NEVER enough
    country_found = [
        c for c in COUNTRY_CONTEXT
        if c in t
    ]

    # Non-defence topic indicators
    non_defence = [
        x for x in NON_DEFENCE_TOPICS
        if x in t
    ]

    # --------------------------------------------------------
    # RULE 1:
    # Strong defence phrase -> candidate
    # --------------------------------------------------------

    if signals:
        score = 5 + min(len(strong_found), 5)

    # --------------------------------------------------------
    # RULE 2:
    # Multiple independent military terms
    # --------------------------------------------------------

    elif len(strong_found) >= 3:
        score = len(strong_found) + 1

    # --------------------------------------------------------
    # RULE 3:
    # Only one military word is NOT enough
    # --------------------------------------------------------

    else:
        return False, [], 0

    # Penalise obviously unrelated articles
    if non_defence:
        # Cultural article containing "army" etc.
        if not signals and len(strong_found) <= 4:
            return False, [], 0

        # If article is overwhelmingly non-defence,
        # reject unless it contains a very explicit military event.
        non_defence_count = sum(
            1 for x in non_defence if x in t
        )

        if non_defence_count >= 3 and not signals:
            return False, [], 0

    # Need actual military context
    if not signals and len(strong_found) < 3:
        return False, [], 0

    final_signals = list(dict.fromkeys(
        signals + strong_found[:8]
    ))

    return True, final_signals, score


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_native_blocks(page):
    """
    Extract text + coordinates directly from text-based PDFs.

    This is much better than rasterising a text PDF and OCRing it.
    """

    data = page.get_text("dict")

    blocks = []

    for block in data.get("blocks", []):

        if block.get("type") != 0:
            continue

        bbox = block.get("bbox")

        if not bbox:
            continue

        x0, y0, x1, y1 = bbox

        lines = []

        for line in block.get("lines", []):

            line_text = ""

            sizes = []

            for span in line.get("spans", []):
                txt = span.get("text", "")
                line_text += txt

                if span.get("size"):
                    sizes.append(span.get("size"))

            line_text = clean_line(line_text)

            if line_text:
                lines.append({
                    "text": line_text,
                    "size": max(sizes) if sizes else 0,
                    "bbox": line.get("bbox", bbox)
                })

        if lines:
            blocks.append({
                "bbox": bbox,
                "lines": lines,
                "source": "native"
            })

    return blocks


# ============================================================
# IMAGE OCR
# ============================================================

def render_page(page, zoom=2.5):

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.open(
        io.BytesIO(pix.tobytes("png"))
    ).convert("RGB")

    return image


def preprocess_image(image):

    gray = ImageOps.grayscale(image)

    # Improve contrast
    gray = ImageEnhance.Contrast(gray).enhance(1.5)

    # Slight sharpening
    gray = gray.filter(ImageFilter.SHARPEN)

    return gray


def ocr_blocks(image):

    processed = preprocess_image(image)

    data = pytesseract.image_to_data(
        processed,
        config=OCR_CONFIG,
        lang="eng",
        output_type=pytesseract.Output.DICT
    )

    grouped = defaultdict(list)

    n = len(data["text"])

    for i in range(n):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except:
            conf = 0

        if conf < 15:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        grouped[key].append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "conf": conf
        })

    blocks = []

    for key, words in grouped.items():

        if not words:
            continue

        words.sort(key=lambda x: x["left"])

        text = " ".join(
            x["text"] for x in words
        )

        left = min(x["left"] for x in words)
        top = min(x["top"] for x in words)

        right = max(
            x["left"] + x["width"]
            for x in words
        )

        bottom = max(
            x["top"] + x["height"]
            for x in words
        )

        avg_height = sum(
            x["height"] for x in words
        ) / len(words)

        blocks.append({
            "bbox": (
                left,
                top,
                right,
                bottom
            ),
            "lines": [{
                "text": clean_line(text),
                "size": avg_height,
                "bbox": (
                    left,
                    top,
                    right,
                    bottom
                )
            }],
            "source": "ocr"
        })

    return blocks


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(blocks, page_width):

    if not blocks:
        return [(0, page_width)]

    # Ignore very wide blocks because they may be:
    # page title / newspaper masthead / full-width headline.
    normal_blocks = []

    for b in blocks:

        x0, y0, x1, y1 = b["bbox"]

        width = x1 - x0

        if width < page_width * 0.75:
            normal_blocks.append(b)

    if len(normal_blocks) < 3:
        return [(0, page_width)]

    # Collect left edges
    xs = sorted(
        b["bbox"][0]
        for b in normal_blocks
    )

    # Cluster x positions
    clusters = []

    tolerance = page_width * 0.035

    for x in xs:

        placed = False

        for cluster in clusters:

            if abs(
                x - sum(cluster) / len(cluster)
            ) <= tolerance:

                cluster.append(x)
                placed = True
                break

        if not placed:
            clusters.append([x])

    clusters = [
        c for c in clusters
        if len(c) >= 2
    ]

    if len(clusters) < 2:
        return [(0, page_width)]

    centers = sorted(
        sum(c) / len(c)
        for c in clusters
    )

    # Maximum 5 newspaper columns
    centers = centers[:5]

    boundaries = [0]

    for i in range(len(centers) - 1):

        mid = (
            centers[i] + centers[i + 1]
        ) / 2

        boundaries.append(mid)

    boundaries.append(page_width)

    columns = []

    for i in range(len(boundaries) - 1):

        x0 = boundaries[i]
        x1 = boundaries[i + 1]

        if x1 - x0 > page_width * 0.08:
            columns.append((x0, x1))

    return columns


# ============================================================
# ASSIGN BLOCKS TO COLUMNS
# ============================================================

def blocks_in_column(blocks, column):

    cx0, cx1 = column

    selected = []

    for b in blocks:

        x0, y0, x1, y1 = b["bbox"]

        center = (x0 + x1) / 2

        if cx0 <= center <= cx1:
            selected.append(b)

    selected.sort(
        key=lambda b: (
            b["bbox"][1],
            b["bbox"][0]
        )
    )

    return selected


# ============================================================
# HEADLINE DETECTION
# ============================================================

def looks_like_headline(block):

    text = clean_line(
        " ".join(
            x["text"]
            for x in block["lines"]
        )
    )

    if not text:
        return False

    wc = word_count(text)

    if wc > 35:
        return False

    # Native PDF font size
    sizes = [
        x.get("size", 0)
        for x in block["lines"]
    ]

    avg_size = (
        sum(sizes) / len(sizes)
        if sizes else 0
    )

    if block["source"] == "native":

        if avg_size >= 15:
            return True

    else:

        # OCR text height
        if avg_size >= 18 and wc <= 20:
            return True

    # ALL CAPS headline
    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if (
        len(letters) > 8
        and letters.upper() == letters
    ):
        return True

    # Common newspaper headline shape
    if wc <= 14 and len(text) <= 110:
        if not text.endswith("."):
            return True

    return False


# ============================================================
# BLOCK TEXT
# ============================================================

def block_text(block):

    return clean_line(
        " ".join(
            x["text"]
            for x in block["lines"]
        )
    )


# ============================================================
# ARTICLE GROUPING
# ============================================================

def build_article_candidates(column_blocks):

    if not column_blocks:
        return []

    articles = []
    current = []

    previous_bottom = None

    for index, block in enumerate(column_blocks):

        text = block_text(block)

        if not text:
            continue

        x0, y0, x1, y1 = block["bbox"]

        is_headline = looks_like_headline(block)

        if previous_bottom is None:
            current.append(block)

        else:

            gap = y0 - previous_bottom

            # Large vertical gap strongly suggests
            # separate newspaper content.
            large_gap = gap > max(
                35,
                (y1 - y0) * 1.8
            )

            # Headline after an existing article
            # normally starts a new article.
            new_headline = (
                is_headline
                and len(current) > 1
            )

            # Very large gap = definite boundary
            if large_gap and len(current) > 0:
                articles.append(current)
                current = [block]

            elif new_headline:
                articles.append(current)
                current = [block]

            else:
                current.append(block)

        previous_bottom = max(
            previous_bottom or 0,
            y1
        )

    if current:
        articles.append(current)

    return articles


# ============================================================
# ARTICLE RECONSTRUCTION
# ============================================================

def reconstruct_article(blocks):

    if not blocks:
        return ""

    # Sort by vertical position.
    blocks = sorted(
        blocks,
        key=lambda b: (
            b["bbox"][1],
            b["bbox"][0]
        )
    )

    lines = []

    for block in blocks:

        text = block_text(block)

        if text:
            lines.append(text)

    # Remove accidental duplicate lines
    result = []

    seen = set()

    for line in lines:

        key = re.sub(
            r"[^a-z0-9]",
            "",
            line.lower()
        )

        if len(key) < 5:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(line)

    return "\n".join(result).strip()


# ============================================================
# ARTICLE CLEANUP
# ============================================================

def clean_article(text):

    text = normalize_text(text)

    lines = []

    for line in text.splitlines():

        line = clean_line(line)

        if not line:
            continue

        # Remove common newspaper continuation markers
        if re.fullmatch(
            r"(continued on page \d+|page \d+)",
            line.lower()
        ):
            continue

        lines.append(line)

    # Remove repeated consecutive lines
    final = []

    for line in lines:

        if final:
            if line.lower() == final[-1].lower():
                continue

        final.append(line)

    return "\n".join(final)


# ============================================================
# ARTICLE QUALITY
# ============================================================

def article_quality(text):

    wc = word_count(text)

    if wc < MIN_ARTICLE_WORDS:
        return False

    # OCR junk ratio
    chars = len(text)

    if chars < 100:
        return False

    alpha = sum(
        c.isalpha()
        for c in text
    )

    if alpha / max(chars, 1) < 0.55:
        return False

    # If almost every line is tiny, likely OCR fragments
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    if len(lines) >= 8:

        short_lines = sum(
            1 for x in lines
            if word_count(x) <= 2
        )

        if short_lines / len(lines) > 0.55:
            return False

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def fingerprint(text):

    normalized = re.sub(
        r"[^a-z0-9]",
        "",
        text.lower()
    )

    return hashlib.md5(
        normalized[:5000].encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


def remove_duplicates(articles):

    result = []
    seen = set()

    for article in articles:

        fp = fingerprint(
            article["text"]
        )

        if fp in seen:
            continue

        seen.add(fp)
        result.append(article)

    return result


# ============================================================
# PAGE PROCESSING
# ============================================================

def process_page(page, page_number):

    page_rect = page.rect

    page_width = page_rect.width
    page_height = page_rect.height

    # --------------------------------------------------------
    # FIRST: native PDF text
    # --------------------------------------------------------

    native_blocks = extract_native_blocks(page)

    native_words = sum(
        word_count(block_text(b))
        for b in native_blocks
    )

    # If PDF has useful text, use it.
    # Otherwise OCR the page.
    if native_words >= 40:

        blocks = native_blocks
        extraction_method = "Native PDF"

    else:

        image = render_page(page)

        blocks = ocr_blocks(image)
        extraction_method = "OCR"

        # Scale coordinates back approximately
        # to PDF page coordinates
        scale_x = page_width / image.width
        scale_y = page_height / image.height

        for b in blocks:

            x0, y0, x1, y1 = b["bbox"]

            b["bbox"] = (
                x0 * scale_x,
                y0 * scale_y,
                x1 * scale_x,
                y1 * scale_y
            )

            for line in b["lines"]:

                lx0, ly0, lx1, ly1 = line["bbox"]

                line["bbox"] = (
                    lx0 * scale_x,
                    ly0 * scale_y,
                    lx1 * scale_x,
                    ly1 * scale_y
                )

    if not blocks:
        return [], extraction_method

    # --------------------------------------------------------
    # DETECT COLUMNS
    # --------------------------------------------------------

    columns = detect_columns(
        blocks,
        page_width
    )

    page_articles = []

    # --------------------------------------------------------
    # PROCESS EACH COLUMN SEPARATELY
    # --------------------------------------------------------

    for column_number, column in enumerate(columns, start=1):

        col_blocks = blocks_in_column(
            blocks,
            column
        )

        if not col_blocks:
            continue

        candidate_groups = build_article_candidates(
            col_blocks
        )

        for candidate in candidate_groups:

            text = reconstruct_article(
                candidate
            )

            text = clean_article(text)

            if not article_quality(text):
                continue

            is_defence, signals, score = defence_analysis(
                text
            )

            if not is_defence:
                continue

            page_articles.append({
                "page": page_number,
                "column": column_number,
                "text": text,
                "signals": signals,
                "score": score,
                "method": extraction_method,
            })

    return page_articles, extraction_method


# ============================================================
# MAIN PDF SCANNER
# ============================================================

def scan_pdf(uploaded_file, page_numbers, progress_bar=None):

    pdf_bytes = uploaded_file.getvalue()

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    all_articles = []

    native_count = 0
    ocr_count = 0

    total = len(page_numbers)

    for position, page_number in enumerate(page_numbers):

        page = doc[page_number]

        articles, method = process_page(
            page,
            page_number + 1
        )

        all_articles.extend(articles)

        if method == "Native PDF":
            native_count += 1
        else:
            ocr_count += 1

        if progress_bar:

            progress_bar.progress(
                int(
                    ((position + 1) / total) * 100
                )
            )

    doc.close()

    # --------------------------------------------------------
    # GLOBAL DEDUPLICATION
    # --------------------------------------------------------

    all_articles = remove_duplicates(
        all_articles
    )

    # Highest confidence first
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
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Defence News Scanner OCR")

st.caption(
    "Layout-aware newspaper OCR for complete defence-related news articles"
)

uploaded_file = st.file_uploader(
    "Upload Newspaper PDF",
    type=["pdf"]
)

if uploaded_file:

    try:

        doc = fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        )

        total_pages = len(doc)

        doc.close()

        st.info(
            f"📄 {uploaded_file.name} | "
            f"{total_pages} pages"
        )

        st.subheader("Select scan range")

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

            page_numbers = list(
                range(
                    min(3, total_pages)
                )
            )

        elif scan_mode == "All pages":

            page_numbers = list(
                range(total_pages)
            )

        else:

            selected = st.multiselect(
                "Select pages",
                options=list(
                    range(
                        1,
                        total_pages + 1
                    )
                ),
                default=[1]
            )

            page_numbers = [
                p - 1
                for p in selected
            ]

        st.write(
            f"**Pages selected:** {len(page_numbers)}"
        )

        if st.button(
            "🔎 Scan Newspaper",
            type="primary",
            disabled=len(page_numbers) == 0
        ):

            progress = st.progress(0)

            with st.spinner(
                "Analysing newspaper layout and extracting complete articles..."
            ):

                articles, native_count, ocr_count = scan_pdf(
                    uploaded_file,
                    page_numbers,
                    progress
                )

            progress.progress(100)

            st.success(
                f"Scan completed — "
                f"{len(articles)} defence article(s) found."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Pages scanned",
                len(page_numbers)
            )

            c2.metric(
                "Native PDF",
                native_count
            )

            c3.metric(
                "OCR",
                ocr_count
            )

            st.divider()

            st.subheader("📰 Defence News Found")

            if not articles:

                st.warning(
                    "No complete defence-related article "
                    "passed the strict relevance filter."
                )

            else:

                for i, article in enumerate(
                    articles,
                    start=1
                ):

                    with st.container(
                        border=True
                    ):

                        st.markdown(
                            f"### News {i}"
                        )

                        st.caption(
                            f"📄 Page {article['page']} • "
                            f"Column {article['column']} • "
                            f"{article['method']}"
                        )

                        st.write(
                            "🔎 **Defence signals:** "
                            + ", ".join(
                                article["signals"]
                            )
                        )

                        st.markdown(
                            article["text"]
                        )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            if articles:

                output_parts = []

                for i, article in enumerate(
                    articles,
                    start=1
                ):

                    output_parts.append(
                        f"NEWS {i}\n"
                        f"Page: {article['page']}\n"
                        f"Column: {article['column']}\n"
                        f"Signals: "
                        f"{', '.join(article['signals'])}\n\n"
                        f"{article['text']}\n"
                        f"\n{'=' * 80}\n"
                    )

                output_text = "\n".join(
                    output_parts
                )

                st.download_button(
                    "⬇️ Download Extracted Defence News",
                    data=output_text,
                    file_name="defence_news_extracted.txt",
                    mime="text/plain"
                )

    except Exception as e:

        st.error(
            "Scanner error:"
        )

        st.exception(e)
