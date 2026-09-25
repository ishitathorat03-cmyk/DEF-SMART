import streamlit as st
import fitz
import re
import hashlib
import statistics
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

st.title("🛡️ Defence News Scanner OCR")
st.caption(
    "Layout-aware newspaper OCR for complete defence-related news articles"
)


# ============================================================
# STRICT DEFENCE VOCABULARY
# ============================================================

# These are strong defence signals.
STRONG_DEFENCE = {
    "army",
    "navy",
    "air force",
    "armed forces",
    "military",
    "soldier",
    "soldiers",
    "troop",
    "troops",
    "militant",
    "militants",
    "regiment",
    "battalion",
    "brigade",
    "commander",
    "commanders",
    "corps commander",
    "military commander",
    "defence ministry",
    "defense ministry",
    "defence minister",
    "defense minister",
    "military operation",
    "military operations",
    "operation",
    "missile",
    "missiles",
    "rocket",
    "rockets",
    "drone strike",
    "drone strikes",
    "airstrike",
    "airstrikes",
    "air strike",
    "air strikes",
    "fighter jet",
    "fighter jets",
    "fighter aircraft",
    "warship",
    "warships",
    "submarine",
    "submarines",
    "artillery",
    "ammunition",
    "weapon",
    "weapons",
    "defence procurement",
    "defense procurement",
    "military aircraft",
    "military aircrafts",
    "air defence",
    "air defense",
    "border security",
    "border forces",
    "special forces",
    "paramilitary",
    "counter-terror",
    "counterterror",
    "counter-terrorism",
    "terrorist attack",
    "terrorist attacks",
    "terror attack",
    "terror attacks",
    "ceasefire",
    "war",
    "warfare",
    "combat",
    "combat operation",
    "combat operations",
    "military exercise",
    "military exercises",
    "defence deal",
    "defence agreement",
    "defense deal",
    "defense agreement",
    "military agreement",
    "military pact",
    "military deployment",
    "troop deployment",
    "military deployment",
    "line of actual control",
    "lac",
    "line of control",
    "loc",
    "border clash",
    "border clashes",
    "military talks",
    "army chief",
    "navy chief",
    "air chief",
    "air chief marshal",
    "chief of army staff",
    "chief of defence staff",
    "chief of defense staff",
    "military aircraft",
    "helicopter",
    "helicopters",
    "fighter",
    "fighters",
    "tank",
    "tanks",
    "artillery",
    "rifle",
    "rifles",
    "gunship",
    "aircraft carrier",
    "aircraft carriers",
    "frigate",
    "frigates",
    "destroyer",
    "destroyers",
    "cruise missile",
    "ballistic missile",
    "ballistic missiles",
    "nuclear missile",
    "nuclear missiles",
    "warplane",
    "warplanes",
}

# These are contextual only.
# They NEVER qualify an article by themselves.
CONTEXT_TERMS = {
    "india",
    "china",
    "pakistan",
    "russia",
    "ukraine",
    "iran",
    "israel",
    "nato",
    "pentagon",
    "kremlin",
    "zelenskyy",
    "zelensky",
    "lavrov",
    "gaza",
    "lebanon",
    "syria",
    "afghanistan",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x0c", " ")
    text = text.replace("|", " ")

    # Join words broken by line hyphenation.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def normalize(text):
    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def words(text):
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())


# ============================================================
# STRICT DEFENCE CLASSIFICATION
# ============================================================

def defence_analysis(text):
    """
    Very strict article-level classifier.

    Important:
    - Country names alone do NOT qualify.
    - 'army' in a historical/cultural article is not enough.
    - We require strong defence context.
    """

    t = normalize(text)

    strong_matches = []
    contextual_matches = []

    # Long phrases first.
    for term in sorted(STRONG_DEFENCE, key=len, reverse=True):
        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, t):
            strong_matches.append(term)

    for term in sorted(CONTEXT_TERMS, key=len, reverse=True):
        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, t):
            contextual_matches.append(term)

    # Remove duplicates.
    strong_matches = list(dict.fromkeys(strong_matches))
    contextual_matches = list(dict.fromkeys(contextual_matches))

    # --------------------------------------------------------
    # HARD EXCLUSIONS
    # --------------------------------------------------------

    # A single weak military word should not qualify an article.
    if len(strong_matches) == 1:

        only = strong_matches[0]

        weak_single_terms = {
            "army",
            "soldier",
            "soldiers",
            "commander",
            "war",
            "operation",
            "fighter",
            "helicopter",
            "tank",
        }

        if only in weak_single_terms:
            return {
                "is_defence": False,
                "score": 0,
                "strong": strong_matches,
                "context": contextual_matches,
            }

    # --------------------------------------------------------
    # STRONG PHRASES
    # --------------------------------------------------------

    very_strong = {
        "air defence",
        "air defense",
        "defence ministry",
        "defense ministry",
        "defence procurement",
        "defense procurement",
        "military operation",
        "military operations",
        "military commander",
        "military deployment",
        "military exercise",
        "military exercises",
        "ballistic missile",
        "cruise missile",
        "aircraft carrier",
        "special forces",
        "chief of army staff",
        "chief of defence staff",
        "chief of defense staff",
        "line of actual control",
        "line of control",
        "border security",
        "border clashes",
        "military talks",
        "defence agreement",
        "defense agreement",
        "military agreement",
        "military pact",
        "counter-terrorism",
        "terrorist attack",
        "terrorist attacks",
        "drone strike",
        "drone strikes",
        "airstrike",
        "airstrikes",
    }

    has_very_strong = any(term in t for term in very_strong)

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    # Very strong phrase is enough.
    if has_very_strong:
        return {
            "is_defence": True,
            "score": 5,
            "strong": strong_matches,
            "context": contextual_matches,
        }

    # Multiple independent strong signals.
    #
    # Example:
    # "army + corps commander + LAC" -> YES
    #
    # But:
    # "army officer + Bharatanatyam" -> NO
    if len(strong_matches) >= 2:
        return {
            "is_defence": True,
            "score": min(5, len(strong_matches)),
            "strong": strong_matches,
            "context": contextual_matches,
        }

    return {
        "is_defence": False,
        "score": 0,
        "strong": strong_matches,
        "context": contextual_matches,
    }


# ============================================================
# PDF PAGE RENDERING
# ============================================================

def render_page(page):
    matrix = fitz.Matrix(2.0, 2.0)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return img


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image):
    img = image.convert("L")

    # Make OCR easier.
    target_width = 2200

    if img.width < target_width:
        ratio = target_width / img.width
        img = img.resize(
            (
                target_width,
                int(img.height * ratio)
            )
        )

    img = ImageEnhance.Contrast(img).enhance(1.35)

    img = ImageEnhance.Sharpness(img).enhance(1.25)

    img = img.filter(ImageFilter.SHARPEN)

    return img


# ============================================================
# FIRST OCR PASS
# Detect newspaper columns
# ============================================================

def get_words_with_boxes(image):

    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 3",
        output_type=Output.DICT
    )

    result = []

    n = len(data["text"])

    for i in range(n):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except:
            conf = -1

        if conf < 20:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        result.append({
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + w / 2,
            "cy": y + h / 2,
            "block": data["block_num"][i],
            "par": data["par_num"][i],
            "line": data["line_num"][i],
        })

    return result


def detect_columns(words_data, image_width):

    if not words_data:
        return [(0, image_width)]

    centers = sorted(
        [w["cx"] for w in words_data]
    )

    if len(centers) < 30:
        return [(0, image_width)]

    gaps = []

    for i in range(1, len(centers)):
        gap = centers[i] - centers[i - 1]

        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return [(0, image_width)]

    median_gap = statistics.median(gaps)

    # Newspaper column gap is normally much larger than
    # ordinary word-to-word spacing.
    threshold = max(
        45,
        median_gap * 6,
        image_width * 0.025
    )

    possible = []

    for i in range(1, len(centers)):

        gap = centers[i] - centers[i - 1]

        if gap >= threshold:

            left = centers[i - 1]
            right = centers[i]

            # Avoid treating tiny gaps as column boundaries.
            if right - left >= image_width * 0.04:

                possible.append(
                    (
                        gap,
                        (left + right) / 2
                    )
                )

    possible.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    # Maximum 4 newspaper columns.
    selected = []

    min_distance = image_width * 0.10

    for gap, center in possible:

        if all(
            abs(center - c) > min_distance
            for c in selected
        ):
            selected.append(center)

        if len(selected) >= 3:
            break

    if not selected:
        return [(0, image_width)]

    selected.sort()

    boundaries = [0]

    for center in selected:
        boundaries.append(int(center))

    boundaries.append(image_width)

    columns = []

    for i in range(len(boundaries) - 1):

        x0 = boundaries[i]
        x1 = boundaries[i + 1]

        width = x1 - x0

        if width >= image_width * 0.15:
            columns.append((x0, x1))

    if len(columns) <= 1:
        return [(0, image_width)]

    return columns


# ============================================================
# COLUMN OCR
# ============================================================

def ocr_column(image, x0, x1):

    # Small overlap helps avoid cutting characters.
    pad = 12

    crop_x0 = max(0, x0 - pad)
    crop_x1 = min(image.width, x1 + pad)

    crop = image.crop(
        (
            crop_x0,
            0,
            crop_x1,
            image.height
        )
    )

    data = pytesseract.image_to_data(
        crop,
        config="--oem 3 --psm 4",
        output_type=Output.DICT
    )

    lines = {}

    n = len(data["text"])

    for i in range(n):

        txt = data["text"][i].strip()

        if not txt:
            continue

        try:
            conf = float(data["conf"][i])
        except:
            conf = -1

        if conf < 20:
            continue

        block = data["block_num"][i]
        par = data["par_num"][i]
        line = data["line_num"][i]

        key = (
            block,
            par,
            line
        )

        if key not in lines:
            lines[key] = {
                "words": [],
                "top": int(data["top"][i]),
                "bottom": int(data["top"][i])
                       + int(data["height"][i]),
                "height": int(data["height"][i]),
            }

        top = int(data["top"][i])
        bottom = top + int(data["height"][i])

        lines[key]["top"] = min(
            lines[key]["top"],
            top
        )

        lines[key]["bottom"] = max(
            lines[key]["bottom"],
            bottom
        )

        lines[key]["height"] = max(
            lines[key]["height"],
            int(data["height"][i])
        )

        lines[key]["words"].append(
            (
                int(data["left"][i]),
                txt
            )
        )

    result = []

    for info in lines.values():

        info["words"].sort(
            key=lambda x: x[0]
        )

        text = " ".join(
            word
            for _, word in info["words"]
        )

        text = clean_text(text)

        if not text:
            continue

        result.append({
            "text": text,
            "top": info["top"],
            "bottom": info["bottom"],
            "height": info["height"],
        })

    result.sort(
        key=lambda x: x["top"]
    )

    return result


# ============================================================
# HEADLINE DETECTION
# ============================================================

def looks_like_headline(line, median_height):

    text = clean_text(line["text"])

    if not text:
        return False

    wc = len(text.split())

    # Very large text compared with ordinary body text.
    if median_height > 0:

        ratio = line["height"] / median_height

        if ratio >= 1.55 and wc <= 25:
            return True

        if ratio >= 1.30 and wc <= 12:
            return True

    # Newspaper headlines often have short uppercase text.
    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if len(letters) >= 5:

        upper_ratio = sum(
            c.isupper()
            for c in letters
        ) / len(letters)

        if upper_ratio >= 0.85 and wc <= 14:
            return True

    # Headline-like punctuation patterns.
    if wc <= 12 and (
        ":" in text
        or "?" in text
        or "—" in text
    ):
        return True

    return False


# ============================================================
# BUILD ARTICLE BLOCKS
# ============================================================

def build_article_blocks(lines):

    if not lines:
        return []

    heights = [
        line["height"]
        for line in lines
        if line["height"] > 0
    ]

    median_height = (
        statistics.median(heights)
        if heights
        else 15
    )

    articles = []

    current = []

    for i, line in enumerate(lines):

        text = clean_text(line["text"])

        if not text:
            continue

        is_head = looks_like_headline(
            line,
            median_height
        )

        if not current:

            current = [line]
            continue

        previous = current[-1]

        gap = (
            line["top"]
            - previous["bottom"]
        )

        # A large vertical gap strongly indicates
        # a new newspaper story.
        large_gap = (
            gap > max(
                28,
                median_height * 2.4
            )
        )

        # New headline after existing body.
        if is_head and len(current) >= 2:

            articles.append(current)

            current = [line]

            continue

        # Extremely large gap = new story.
        if large_gap and len(current) >= 2:

            articles.append(current)

            current = [line]

            continue

        current.append(line)

    if current:
        articles.append(current)

    return articles


# ============================================================
# ARTICLE CLEANING
# ============================================================

def article_text(lines):

    text_lines = []

    for line in lines:

        text = clean_text(line["text"])

        if not text:
            continue

        text_lines.append(text)

    if not text_lines:
        return ""

    # Fix common OCR spaces.
    text = "\n".join(text_lines)

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text
    )

    text = re.sub(
        r"([(\[])\s+",
        r"\1",
        text
    )

    text = re.sub(
        r"\s+([)\]])",
        r"\1",
        text
    )

    return clean_text(text)


# ============================================================
# ARTICLE TITLE / BODY QUALITY
# ============================================================

def article_quality(text):

    wc = len(text.split())

    if wc < 35:
        return False

    # Reject obvious navigation fragments.
    bad_patterns = [
        r"^page\s+\d+$",
        r"^continued on page",
        r"^see page",
        r"^advertisement$",
    ]

    low = normalize(text)

    for p in bad_patterns:

        if re.search(p, low):
            return False

    # A genuine article generally has several sentences.
    sentence_count = len(
        re.findall(
            r"[.!?](?:\s|$)",
            text
        )
    )

    if sentence_count < 2 and wc < 60:
        return False

    return True


# ============================================================
# STRICT ARTICLE EXTRACTION
# ============================================================

def extract_defence_articles(lines):

    blocks = build_article_blocks(lines)

    results = []

    for block_index, block in enumerate(blocks):

        text = article_text(block)

        if not article_quality(text):
            continue

        analysis = defence_analysis(text)

        if not analysis["is_defence"]:
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # Do not return just the sentence containing
        # "missile", "army", etc.
        #
        # Return the COMPLETE newspaper block.
        # ----------------------------------------------------

        results.append({
            "text": text,
            "score": analysis["score"],
            "strong": analysis["strong"],
            "context": analysis["context"],
            "block": block_index,
        })

    return results


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def fingerprint(text):

    t = normalize(text)

    t = re.sub(
        r"[^a-z0-9 ]",
        "",
        t
    )

    t = re.sub(
        r"\s+",
        " ",
        t
    )

    return hashlib.md5(
        t.encode("utf-8")
    ).hexdigest()


def similarity_key(text):

    tokens = normalize(text).split()

    if len(tokens) > 60:
        tokens = tokens[:60]

    return " ".join(tokens)


def remove_duplicates(articles):

    final = []

    seen_hashes = set()
    seen_keys = set()

    for article in articles:

        text = article["text"]

        h = fingerprint(text)

        key = similarity_key(text)

        if h in seen_hashes:
            continue

        # Similar beginning = likely duplicate caused by
        # column overlap.
        duplicate = False

        for old in seen_keys:

            if key[:120] == old[:120]:
                duplicate = True
                break

        if duplicate:
            continue

        seen_hashes.add(h)
        seen_keys.add(key)

        final.append(article)

    return final


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(page):

    image = render_page(page)

    image = preprocess(image)

    # First OCR pass only for column detection.
    words_data = get_words_with_boxes(image)

    columns = detect_columns(
        words_data,
        image.width
    )

    page_articles = []

    for column_number, (x0, x1) in enumerate(columns, start=1):

        lines = ocr_column(
            image,
            x0,
            x1
        )

        if not lines:
            continue

        articles = extract_defence_articles(
            lines
        )

        for article in articles:

            article["column"] = column_number
            article["x0"] = x0
            article["x1"] = x1

            page_articles.append(article)

    page_articles = remove_duplicates(
        page_articles
    )

    return page_articles, len(columns)


# ============================================================
# FILE HASH
# ============================================================

def file_hash(data):

    return hashlib.md5(data).hexdigest()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload Newspaper PDF",
    type=["pdf"]
)


if uploaded:

    pdf_bytes = uploaded.read()

    current_hash = file_hash(
        pdf_bytes
    )

    try:

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        st.success(
            f"📄 {uploaded.name} | {len(doc)} pages"
        )

    except Exception as e:

        st.error(
            f"Could not open PDF: {e}"
        )

        st.stop()

    # ========================================================
    # SCAN RANGE
    # ========================================================

    st.subheader("Select scan range")

    mode = st.radio(
        "",
        [
            "First 3 pages - TEST",
            "All pages",
            "Select pages"
        ],
        horizontal=True
    )

    if mode == "First 3 pages - TEST":

        page_numbers = list(
            range(
                min(3, len(doc))
            )
        )

    elif mode == "All pages":

        page_numbers = list(
            range(len(doc))
        )

    else:

        selected = st.multiselect(
            "Select pages",
            options=list(
                range(
                    1,
                    len(doc) + 1
                )
            ),
            default=[1]
        )

        page_numbers = [
            x - 1
            for x in selected
        ]

    # ========================================================
    # SCAN
    # ========================================================

    if st.button(
        "🔎 Scan Defence News",
        type="primary"
    ):

        if not page_numbers:

            st.warning(
                "Please select at least one page."
            )

            st.stop()

        st.session_state.results = []

        progress = st.progress(0)

        status = st.empty()

        all_results = []

        for count, page_index in enumerate(
            page_numbers,
            start=1
        ):

            status.write(
                f"🔍 Scanning page {page_index + 1} "
                f"({count}/{len(page_numbers)})..."
            )

            try:

                page = doc.load_page(
                    page_index
                )

                articles, column_count = process_page(
                    page
                )

                for article in articles:

                    article["page"] = (
                        page_index + 1
                    )

                    article["column_count"] = (
                        column_count
                    )

                    all_results.append(
                        article
                    )

            except Exception as e:

                st.warning(
                    f"Page {page_index + 1} failed: {e}"
                )

            progress.progress(
                count / len(page_numbers)
            )

        # ----------------------------------------------------
        # FINAL DUPLICATE REMOVAL
        # ----------------------------------------------------

        final_results = []

        seen = set()

        for article in all_results:

            h = fingerprint(
                article["text"]
            )

            if h in seen:
                continue

            seen.add(h)

            final_results.append(
                article
            )

        st.session_state.results = final_results

        status.success(
            f"Scan completed — "
            f"{len(final_results)} defence articles found."
        )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results

if results:

    st.divider()

    st.header(
        "📰 Defence News Found"
    )

    st.caption(
        "Only articles passing the strict defence relevance filter are shown."
    )

    for i, article in enumerate(
        results,
        start=1
    ):

        with st.container(
            border=True
        ):

            st.subheader(
                f"News {i}"
            )

            st.caption(
                f"📄 Page {article['page']} "
                f"• Column {article['column']}"
            )

            strong = article.get(
                "strong",
                []
            )

            context = article.get(
                "context",
                []
            )

            if strong:

                st.write(
                    "🔎 Defence signals: "
                    + ", ".join(strong)
                )

            # ------------------------------------------------
            # ACTUAL ARTICLE
            # ------------------------------------------------

            st.markdown(
                article["text"]
            )

            st.divider()


    # ========================================================
    # DOWNLOAD
    # ========================================================

    download_text = ""

    for i, article in enumerate(
        results,
        start=1
    ):

        download_text += (
            f"\n{'=' * 80}\n"
        )

        download_text += (
            f"NEWS {i}\n"
        )

        download_text += (
            f"PAGE: {article['page']} | "
            f"COLUMN: {article['column']}\n"
        )

        download_text += (
            f"{'=' * 80}\n\n"
        )

        download_text += (
            article["text"]
            + "\n\n"
        )

    st.download_button(
        "⬇️ Download Extracted Defence News",
        data=download_text,
        file_name="defence_news_extracted.txt",
        mime="text/plain"
    )

else:

    if uploaded:
        st.info(
            "No defence-related article was detected in the selected pages."
        )
else:
    st.info(
        "Upload a newspaper PDF to begin."
    )
