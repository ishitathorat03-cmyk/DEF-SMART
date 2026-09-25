import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output
import fitz
import pandas as pd
import re
import hashlib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TESSERACT
# ============================================================

# Local Windows Tesseract
pytesseract.pytesseract.tesseract_cmd = "tesseract"


# ============================================================
# DEFENCE VOCABULARY
# ============================================================

DEFENCE_CORE = [
    "indian army",
    "indian navy",
    "indian air force",
    "armed forces",
    "ministry of defence",
    "ministry of defense",
    "defence ministry",
    "defense ministry",
    "military",
    "military operation",
    "military operations",
    "military exercise",
    "military deployment",
    "military strike",
    "military base",
    "military aircraft",
    "military equipment",
    "defence forces",
    "defense forces",
    "army",
    "navy",
    "air force",
]

DEFENCE_EQUIPMENT = [
    "missile",
    "missiles",
    "rocket",
    "rockets",
    "fighter aircraft",
    "fighter jet",
    "fighter jets",
    "warship",
    "warships",
    "submarine",
    "submarines",
    "aircraft carrier",
    "carrier",
    "frigate",
    "destroyer",
    "helicopter",
    "helicopters",
    "drone",
    "drones",
    "uav",
    "artillery",
    "tank",
    "tanks",
    "armoured",
    "armored",
    "ammunition",
    "weapon",
    "weapons",
    "weapon system",
    "weapons system",
    "missile system",
    "air defence",
    "air defense",
]

DEFENCE_OPERATIONS = [
    "operation",
    "operations",
    "deployment",
    "deployed",
    "troops",
    "soldiers",
    "regiment",
    "regiments",
    "battalion",
    "brigade",
    "commando",
    "commandos",
    "special forces",
    "military exercise",
    "joint exercise",
    "naval exercise",
    "army exercise",
    "air force exercise",
    "combat",
    "battle",
    "war",
    "warfare",
    "strike",
    "strikes",
    "airstrike",
    "airstrikes",
]

SECURITY_DEFENCE = [
    "border",
    "border security",
    "cross-border",
    "infiltration",
    "ceasefire",
    "terrorism",
    "terrorist",
    "terrorists",
    "militant",
    "militants",
    "counter-terror",
    "counter terrorism",
    "counterterrorism",
    "insurgency",
    "security forces",
    "paramilitary",
    "bsf",
    "crpf",
    "itbp",
    "cisf",
    "coast guard",
    "line of control",
    "loc",
    "line of actual control",
    "lac",
]

INTERNATIONAL_CONFLICT = [
    "russia",
    "ukraine",
    "nato",
    "israel",
    "iran",
    "gaza",
    "hamas",
    "hezbollah",
    "taiwan",
    "china",
    "north korea",
    "south korea",
    "conflict",
    "armed conflict",
    "military conflict",
    "invasion",
    "war zone",
    "missile strikes",
    "drone strikes",
]

NON_DEFENCE = [
    "stock market",
    "share market",
    "sensex",
    "nifty",
    "real estate",
    "property prices",
    "school admission",
    "college admission",
    "exam result",
    "cricket",
    "football",
    "movie review",
    "film review",
    "celebrity",
    "fashion",
    "recipe",
    "restaurant",
    "wedding",
    "horoscope",
    "shopping",
    "television serial",
    "tv serial",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


# ============================================================
# DEFENCE CLASSIFIER
# ============================================================

def defence_score(text):

    text = normalize_text(text)

    core = sum(1 for x in DEFENCE_CORE if x in text)
    equipment = sum(1 for x in DEFENCE_EQUIPMENT if x in text)
    operations = sum(1 for x in DEFENCE_OPERATIONS if x in text)
    security = sum(1 for x in SECURITY_DEFENCE if x in text)
    international = sum(1 for x in INTERNATIONAL_CONFLICT if x in text)
    negative = sum(1 for x in NON_DEFENCE if x in text)

    score = (
        core * 8
        + equipment * 5
        + operations * 4
        + security * 4
        + international * 3
        - negative * 6
    )

    return score


def is_defence_article(text):

    text = normalize_text(text)

    words = text.split()

    if len(words) < 18:
        return False

    core = sum(1 for x in DEFENCE_CORE if x in text)
    equipment = sum(1 for x in DEFENCE_EQUIPMENT if x in text)
    operations = sum(1 for x in DEFENCE_OPERATIONS if x in text)
    security = sum(1 for x in SECURITY_DEFENCE if x in text)
    international = sum(1 for x in INTERNATIONAL_CONFLICT if x in text)
    negative = sum(1 for x in NON_DEFENCE if x in text)

    # Strong direct defence article
    if core >= 1:
        return True

    # Defence equipment + military context
    if equipment >= 1 and (
        operations >= 1
        or security >= 1
        or core >= 1
    ):
        return True

    # Multiple military equipment terms
    if equipment >= 2:
        return True

    # Border/security article
    if security >= 2:
        return True

    if security >= 1 and operations >= 1:
        return True

    # International conflict + military context
    if international >= 1 and (
        equipment >= 1
        or operations >= 1
        or security >= 1
    ):
        return True

    # War/conflict + military terminology
    if international >= 1 and (
        "war" in text
        or "conflict" in text
        or "military" in text
        or "strike" in text
        or "ceasefire" in text
    ):
        return True

    # Obvious non-defence article
    if negative >= 2 and core == 0 and equipment == 0:
        return False

    return False


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    target_width = 2400

    if width < target_width:

        scale = target_width / width

        image = image.resize(
            (
                int(width * scale),
                int(height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    gray = image.convert("L")

    contrast = ImageEnhance.Contrast(gray)

    gray = contrast.enhance(1.5)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# ============================================================
# OCR BLOCK EXTRACTION
# ============================================================

def get_ocr_blocks(image):

    processed = preprocess_image(image)

    data = pytesseract.image_to_data(
        processed,
        output_type=Output.DICT,
        config="--oem 3 --psm 3"
    )

    blocks = []

    total = len(data["text"])

    for i in range(total):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except:
            confidence = 0

        if confidence < 25:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        block_num = data["block_num"][i]
        par_num = data["par_num"][i]

        blocks.append({
            "x": x,
            "y": y,
            "right": x + w,
            "bottom": y + h,
            "w": w,
            "h": h,
            "text": text,
            "block": block_num,
            "paragraph": par_num
        })

    return blocks


# ============================================================
# GROUP OCR WORDS INTO PARAGRAPH BLOCKS
# ============================================================

def merge_ocr_words_into_blocks(words):

    groups = {}

    for word in words:

        key = (
            word["block"],
            word["paragraph"]
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(word)

    blocks = []

    for key, items in groups.items():

        items = sorted(
            items,
            key=lambda x: (
                x["y"],
                x["x"]
            )
        )

        text = " ".join(
            x["text"]
            for x in items
        )

        if len(text.split()) < 3:
            continue

        x1 = min(x["x"] for x in items)
        y1 = min(x["y"] for x in items)

        x2 = max(x["right"] for x in items)
        y2 = max(x["bottom"] for x in items)

        blocks.append({
            "x": x1,
            "y": y1,
            "right": x2,
            "bottom": y2,
            "w": x2 - x1,
            "h": y2 - y1,
            "text": text
        })

    return blocks


# ============================================================
# COLUMN DETECTION
# ============================================================

def group_into_columns(blocks, page_width):

    if not blocks:
        return []

    blocks = sorted(
        blocks,
        key=lambda b: b["x"]
    )

    columns = []

    tolerance = max(
        70,
        page_width * 0.035
    )

    for block in blocks:

        center = (
            block["x"] +
            block["right"]
        ) / 2

        placed = False

        for column in columns:

            centers = [
                (
                    b["x"] +
                    b["right"]
                ) / 2
                for b in column
            ]

            average_center = sum(centers) / len(centers)

            if abs(center - average_center) <= tolerance:

                column.append(block)

                placed = True

                break

        if not placed:

            columns.append([block])

    columns.sort(
        key=lambda col:
        min(b["x"] for b in col)
    )

    for column in columns:

        column.sort(
            key=lambda b: (
                b["y"],
                b["x"]
            )
        )

    return columns


# ============================================================
# ARTICLE GROUPING
# ============================================================

def build_article_candidates(blocks, page_width):

    if not blocks:
        return []

    columns = group_into_columns(
        blocks,
        page_width
    )

    candidates = []

    for column in columns:

        current = []

        previous = None

        for block in column:

            if previous is None:

                current = [block]

                previous = block

                continue

            vertical_gap = (
                block["y"] -
                previous["bottom"]
            )

            previous_height = max(
                previous["h"],
                15
            )

            # Large vertical separation usually
            # indicates a new article.
            gap_limit = max(
                45,
                previous_height * 2.5
            )

            if vertical_gap > gap_limit:

                if current:

                    candidates.append(
                        current
                    )

                current = [block]

            else:

                current.append(block)

            previous = block

        if current:

            candidates.append(current)

    return candidates


# ============================================================
# JOIN BLOCKS
# ============================================================

def candidate_to_text(candidate):

    candidate = sorted(
        candidate,
        key=lambda b: (
            b["y"],
            b["x"]
        )
    )

    parts = []

    for block in candidate:

        text = clean_text(
            block["text"]
        )

        if text:
            parts.append(text)

    return " ".join(parts)


# ============================================================
# REMOVE VERY OBVIOUS OCR GARBAGE
# ============================================================

def valid_article_text(text):

    text = clean_text(text)

    words = text.split()

    if len(words) < 18:
        return False

    alphabetic = sum(
        1
        for c in text
        if c.isalpha()
    )

    if len(text) == 0:
        return False

    alpha_ratio = alphabetic / len(text)

    if alpha_ratio < 0.55:
        return False

    return True


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def similarity(a, b):

    a_words = set(
        normalize_text(a).split()
    )

    b_words = set(
        normalize_text(b).split()
    )

    if not a_words or not b_words:
        return 0

    intersection = len(
        a_words & b_words
    )

    union = len(
        a_words | b_words
    )

    return intersection / union


def remove_duplicate_articles(articles):

    final = []

    for article in articles:

        duplicate = False

        for existing in final:

            if similarity(
                article,
                existing
            ) >= 0.55:

                duplicate = True
                break

        if not duplicate:

            final.append(article)

    return final


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image_page(image):

    blocks = get_ocr_blocks(image)

    if not blocks:
        return []

    page_width = image.width

    merged_blocks = merge_ocr_words_into_blocks(
        blocks
    )

    candidates = build_article_candidates(
        merged_blocks,
        page_width
    )

    articles = []

    for candidate in candidates:

        text = candidate_to_text(
            candidate
        )

        if not valid_article_text(text):
            continue

        if is_defence_article(text):

            articles.append(text)

    return articles


# ============================================================
# PDF NATIVE TEXT EXTRACTION
# ============================================================

def extract_pdf_native_page(page):

    blocks = page.get_text(
        "blocks"
    )

    result = []

    for block in blocks:

        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        text = clean_text(text)

        if len(text.split()) < 3:
            continue

        result.append({
            "x": x0,
            "y": y0,
            "right": x1,
            "bottom": y1,
            "w": x1 - x0,
            "h": y1 - y0,
            "text": text
        })

    return result


def native_text_is_usable(blocks):

    if not blocks:
        return False

    total_words = sum(
        len(
            b["text"].split()
        )
        for b in blocks
    )

    return total_words >= 50


# ============================================================
# PROCESS NATIVE PDF PAGE
# ============================================================

def process_native_pdf_page(
    page,
    native_blocks
):

    if not native_blocks:
        return []

    page_width = page.rect.width

    candidates = build_article_candidates(
        native_blocks,
        page_width
    )

    articles = []

    for candidate in candidates:

        text = candidate_to_text(
            candidate
        )

        if not valid_article_text(text):
            continue

        if is_defence_article(text):

            articles.append(text)

    return articles


# ============================================================
# PROCESS PDF PAGE
# ============================================================

def process_pdf_page(page):

    native_blocks = extract_pdf_native_page(
        page
    )

    # --------------------------------------------------------
    # FIRST TRY NATIVE PDF TEXT
    # --------------------------------------------------------

    if native_text_is_usable(
        native_blocks
    ):

        articles = process_native_pdf_page(
            page,
            native_blocks
        )

        if articles:

            return articles, "Native PDF"

    # --------------------------------------------------------
    # FALLBACK TO OCR
    # --------------------------------------------------------

    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            1.8,
            1.8
        ),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [
            pix.width,
            pix.height
        ],
        pix.samples
    )

    articles = process_image_page(
        image
    )

    return articles, "OCR"


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(data):

    return hashlib.md5(
        data
    ).hexdigest()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "Extract complete defence-related newspaper articles using OCR and layout-aware analysis."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Newspaper PDF or Image",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    current_hash = get_file_hash(
        file_bytes
    )

    if current_hash != st.session_state.file_hash:

        st.session_state.results = []

        st.session_state.file_hash = current_hash

    file_name = uploaded_file.name

    st.info(
        f"📄 File: {file_name}"
    )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if file_name.lower().endswith(".pdf"):

        pdf = fitz.open(
            stream=file_bytes,
            filetype="pdf"
        )

        total_pages = len(pdf)

        st.write(
            f"📚 Total pages: **{total_pages}**"
        )

        # ----------------------------------------------------
        # PAGE SELECTION
        # ----------------------------------------------------

        page_mode = st.radio(
            "Pages to scan",
            [
                "All pages",
                "Select pages"
            ],
            horizontal=True
        )

        if page_mode == "All pages":

            selected_pages = list(
                range(total_pages)
            )

        else:

            selected_pages = st.multiselect(
                "Select page numbers",
                list(
                    range(
                        1,
                        total_pages + 1
                    )
                ),
                default=list(
                    range(
                        1,
                        min(
                            total_pages,
                            3
                        ) + 1
                    )
                )
            )

            selected_pages = [
                x - 1
                for x in selected_pages
            ]

        # ----------------------------------------------------
        # SCAN BUTTON
        # ----------------------------------------------------

        scan = st.button(
            "🚀 Scan Newspaper",
            type="primary"
        )

        if scan:

            all_articles = []

            progress = st.progress(0)

            status = st.empty()

            native_count = 0
            ocr_count = 0

            for index, page_number in enumerate(
                selected_pages
            ):

                status.info(
                    f"🔎 Analyzing page "
                    f"{page_number + 1} "
                    f"of {total_pages}..."
                )

                page = pdf.load_page(
                    page_number
                )

                try:

                    articles, method = process_pdf_page(
                        page
                    )

                    if method == "Native PDF":
                        native_count += 1
                    else:
                        ocr_count += 1

                    all_articles.extend(
                        articles
                    )

                except Exception as e:

                    st.warning(
                        f"Page {page_number + 1} "
                        f"could not be processed: {e}"
                    )

                progress.progress(
                    (index + 1) /
                    len(selected_pages)
                )

            # ------------------------------------------------
            # FINAL DEDUPLICATION
            # ------------------------------------------------

            all_articles = remove_duplicate_articles(
                all_articles
            )

            st.session_state.results = (
                all_articles
            )

            status.success(
                "✅ Newspaper scanning completed."
            )

            st.write(
                f"📄 Pages processed: "
                f"**{len(selected_pages)}**"
            )

            st.write(
                f"⚡ Native PDF pages: "
                f"**{native_count}**"
            )

            st.write(
                f"🔍 OCR pages: "
                f"**{ocr_count}**"
            )

            st.write(
                f"🛡️ Defence articles found: "
                f"**{len(all_articles)}**"
            )

        pdf.close()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    else:

        scan_image = st.button(
            "🚀 Scan Newspaper Image",
            type="primary"
        )

        if scan_image:

            image = Image.open(
                uploaded_file
            )

            with st.spinner(
                "🔎 Analyzing newspaper image..."
            ):

                articles = process_image_page(
                    image
                )

                articles = remove_duplicate_articles(
                    articles
                )

                st.session_state.results = (
                    articles
                )

            st.success(
                "✅ Image scanning completed."
            )

            st.write(
                f"🛡️ Defence articles found: "
                f"**{len(articles)}**"
            )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results


if results:

    st.divider()

    st.header(
        "📰 Defence News Articles"
    )

    st.success(
        f"{len(results)} complete defence article(s) extracted."
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

            st.write(
                article
            )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    output_text = ""

    for i, article in enumerate(
        results,
        start=1
    ):

        output_text += (
            f"NEWS {i}\n"
            f"{'=' * 80}\n"
            f"{article}\n\n"
        )

    st.download_button(
        "⬇️ Download Defence News",
        data=output_text,
        file_name="defence_news.txt",
        mime="text/plain"
    )

else:

    if uploaded_file:
        st.warning(
            "No defence articles extracted yet. "
            "Click the Scan button after uploading the newspaper."
        )
