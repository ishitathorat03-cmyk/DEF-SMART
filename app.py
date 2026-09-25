import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output
import fitz
import re
import hashlib
from collections import defaultdict


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# TESSERACT
# ============================================================

pytesseract.pytesseract.tesseract_cmd = "tesseract"


# ============================================================
# DEFENCE VOCABULARY
# IMPORTANT:
# Classification happens ONLY AFTER article extraction.
# ============================================================

DIRECT_DEFENCE = [
    "indian army",
    "indian navy",
    "indian air force",
    "armed forces",
    "ministry of defence",
    "ministry of defense",
    "defence ministry",
    "defense ministry",
    "defence forces",
    "defense forces",
    "military forces",
    "military operation",
    "military operations",
    "military exercise",
    "military exercises",
    "military deployment",
    "military strike",
    "military strikes",
    "military base",
    "military aircraft",
    "military equipment",
    "army exercise",
    "naval exercise",
    "air force exercise",
    "joint military exercise",
    "defence procurement",
    "defense procurement",
    "defence deal",
    "defense deal",
    "defence contract",
    "defense contract",
    "defence minister",
    "defense minister",
    "chief of army staff",
    "chief of naval staff",
    "chief of air staff",
    "army chief",
    "navy chief",
    "air chief",
    "chief of defence staff",
    "chief of defense staff",
]

MILITARY_HARDWARE = [
    "missile",
    "missiles",
    "ballistic missile",
    "cruise missile",
    "anti-aircraft",
    "air defence system",
    "air defense system",
    "fighter aircraft",
    "fighter aircraft",
    "fighter jet",
    "fighter jets",
    "warship",
    "warships",
    "submarine",
    "submarines",
    "aircraft carrier",
    "frigate",
    "frigates",
    "destroyer",
    "destroyers",
    "military helicopter",
    "military helicopters",
    "combat helicopter",
    "drone",
    "drones",
    "uav",
    "ucav",
    "artillery",
    "tank",
    "tanks",
    "armoured vehicle",
    "armored vehicle",
    "ammunition",
    "weapons system",
    "weapon system",
    "defence system",
    "defense system",
    "radar",
    "military satellite",
    "fighter aircraft",
]

MILITARY_ACTION = [
    "military strike",
    "air strike",
    "airstrike",
    "airstrikes",
    "missile strike",
    "missile strikes",
    "drone strike",
    "drone strikes",
    "troops",
    "soldiers",
    "military personnel",
    "army personnel",
    "naval personnel",
    "air force personnel",
    "regiment",
    "regiments",
    "battalion",
    "battalions",
    "brigade",
    "brigades",
    "special forces",
    "commando",
    "commandos",
    "combat",
    "combat operations",
    "warfare",
    "military operation",
    "military operations",
    "military deployment",
    "military training",
    "military exercise",
]

SECURITY_HARD = [
    "border security force",
    "border security",
    "border patrol",
    "border force",
    "border guards",
    "bsf",
    "crpf",
    "itbp",
    "cisf",
    "sashastra seema bal",
    "assam rifles",
    "coast guard",
    "indian coast guard",
    "counter terrorism",
    "counter-terrorism",
    "counterterrorism",
    "counter insurgency",
    "counter-insurgency",
    "insurgency",
    "insurgent",
    "insurgents",
    "terrorist attack",
    "terrorist group",
    "terrorist groups",
    "militant group",
    "militant groups",
    "infiltration",
    "cross-border infiltration",
]

INTERNATIONAL_MILITARY = [
    "russia",
    "ukraine",
    "nato",
    "israel",
    "iran",
    "gaza",
    "hamas",
    "hezbollah",
    "taiwan",
    "north korea",
    "south korea",
    "military conflict",
    "armed conflict",
    "invasion",
    "warfare",
]

OBVIOUS_NON_DEFENCE = [
    "stock market",
    "share market",
    "sensex",
    "nifty",
    "stock prices",
    "real estate",
    "property prices",
    "school admission",
    "college admission",
    "exam result",
    "cricket match",
    "football match",
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
    "job market",
    "job openings",
    "artificial intelligence jobs",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize(text):
    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def contains_phrase(text, phrase):
    """
    Exact word/phrase matching.
    Prevents things like 'army' accidentally matching
    unrelated text.
    """
    text = normalize(text)
    phrase = normalize(phrase)

    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"

    return re.search(pattern, text) is not None


def count_matches(text, vocabulary):
    return sum(
        1 for item in vocabulary
        if contains_phrase(text, item)
    )


# ============================================================
# STRICT DEFENCE CLASSIFIER
# ============================================================

def is_defence_article(article):

    text = normalize(article)

    if len(text.split()) < 20:
        return False

    direct = count_matches(text, DIRECT_DEFENCE)
    hardware = count_matches(text, MILITARY_HARDWARE)
    action = count_matches(text, MILITARY_ACTION)
    security = count_matches(text, SECURITY_HARD)
    international = count_matches(text, INTERNATIONAL_MILITARY)
    non_defence = count_matches(text, OBVIOUS_NON_DEFENCE)

    # --------------------------------------------------------
    # VERY STRONG DIRECT DEFENCE ARTICLE
    # --------------------------------------------------------

    if direct >= 1:
        return True

    # --------------------------------------------------------
    # HARDWARE MUST HAVE MILITARY CONTEXT
    # Example:
    # missile + troops
    # submarine + naval
    # drone + military
    # --------------------------------------------------------

    if hardware >= 1 and action >= 1:
        return True

    if hardware >= 1 and security >= 1:
        return True

    # Multiple strong military hardware terms
    if hardware >= 2:
        return True

    # --------------------------------------------------------
    # SECURITY
    # --------------------------------------------------------

    if security >= 2:
        return True

    # --------------------------------------------------------
    # INTERNATIONAL CONFLICT
    #
    # Russia alone = NOT defence
    # Ukraine alone = NOT defence
    # China alone = NOT defence
    #
    # They need military context.
    # --------------------------------------------------------

    if international >= 1 and (
        hardware >= 1
        or action >= 1
        or security >= 1
    ):
        return True

    # --------------------------------------------------------
    # VERY STRONG WAR LANGUAGE
    # --------------------------------------------------------

    strong_war_terms = [
        "missile strike",
        "missile strikes",
        "drone strike",
        "drone strikes",
        "airstrike",
        "airstrikes",
        "military conflict",
        "armed conflict",
        "military operation",
        "military operations",
    ]

    if any(
        contains_phrase(text, x)
        for x in strong_war_terms
    ):
        return True

    # --------------------------------------------------------
    # OBVIOUS NON-DEFENCE OVERRIDE
    # --------------------------------------------------------

    if non_defence >= 2 and direct == 0 and hardware == 0:
        return False

    return False


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    target_width = 2600

    if width < target_width:

        ratio = target_width / width

        image = image.resize(
            (
                int(width * ratio),
                int(height * ratio)
            ),
            Image.Resampling.LANCZOS
        )

    gray = image.convert("L")

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(1.6)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# ============================================================
# OCR WORD DATA
# ============================================================

def ocr_words(image):

    image = preprocess_image(image)

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--oem 3 --psm 3"
    )

    words = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except:
            conf = 0

        if conf < 30:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        words.append({
            "text": text,
            "x": x,
            "y": y,
            "right": x + w,
            "bottom": y + h,
            "w": w,
            "h": h,
            "line": data["line_num"][i],
            "block": data["block_num"][i],
            "paragraph": data["par_num"][i],
        })

    return words


# ============================================================
# OCR PARAGRAPH BLOCKS
# ============================================================

def make_ocr_blocks(words):

    groups = defaultdict(list)

    for word in words:

        key = (
            word["block"],
            word["paragraph"]
        )

        groups[key].append(word)

    blocks = []

    for items in groups.values():

        if not items:
            continue

        items.sort(
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

        x1 = min(
            x["x"] for x in items
        )

        y1 = min(
            x["y"] for x in items
        )

        x2 = max(
            x["right"] for x in items
        )

        y2 = max(
            x["bottom"] for x in items
        )

        blocks.append({
            "x": x1,
            "y": y1,
            "right": x2,
            "bottom": y2,
            "w": x2 - x1,
            "h": y2 - y1,
            "text": text,
            "font_height": sum(
                x["h"] for x in items
            ) / len(items)
        })

    return blocks


# ============================================================
# PDF NATIVE BLOCKS
# ============================================================

def get_pdf_blocks(page):

    blocks = page.get_text(
        "dict"
    ).get("blocks", [])

    result = []

    for block in blocks:

        if block.get("type") != 0:
            continue

        lines = block.get(
            "lines",
            []
        )

        spans = []

        for line in lines:

            for span in line.get(
                "spans",
                []
            ):

                text = span.get(
                    "text",
                    ""
                ).strip()

                if not text:
                    continue

                spans.append({
                    "text": text,
                    "size": span.get(
                        "size",
                        10
                    ),
                    "flags": span.get(
                        "flags",
                        0
                    ),
                })

        if not spans:
            continue

        text = clean(
            " ".join(
                x["text"]
                for x in spans
            )
        )

        if len(text.split()) < 3:
            continue

        x0, y0, x1, y1 = block[
            "bbox"
        ]

        avg_size = sum(
            x["size"]
            for x in spans
        ) / len(spans)

        max_size = max(
            x["size"]
            for x in spans
        )

        result.append({
            "x": x0,
            "y": y0,
            "right": x1,
            "bottom": y1,
            "w": x1 - x0,
            "h": y1 - y0,
            "text": text,
            "font_height": avg_size,
            "max_font": max_size,
        })

    return result


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(blocks, page_width):

    if not blocks:
        return []

    # Newspaper columns are generally separated
    # horizontally. We cluster by x-overlap rather
    # than by article keywords.

    sorted_blocks = sorted(
        blocks,
        key=lambda b: b["x"]
    )

    columns = []

    for block in sorted_blocks:

        placed = False

        for column in columns:

            col_left = min(
                x["x"]
                for x in column
            )

            col_right = max(
                x["right"]
                for x in column
            )

            overlap = max(
                0,
                min(
                    block["right"],
                    col_right
                )
                -
                max(
                    block["x"],
                    col_left
                )
            )

            block_width = max(
                block["w"],
                1
            )

            overlap_ratio = (
                overlap / block_width
            )

            center1 = (
                block["x"]
                +
                block["right"]
            ) / 2

            center2 = (
                col_left
                +
                col_right
            ) / 2

            center_distance = abs(
                center1 - center2
            )

            if (
                overlap_ratio >= 0.30
                or center_distance <= page_width * 0.025
            ):

                column.append(block)
                placed = True
                break

        if not placed:

            columns.append(
                [block]
            )

    columns.sort(
        key=lambda c:
        min(x["x"] for x in c)
    )

    for column in columns:

        column.sort(
            key=lambda x: (
                x["y"],
                x["x"]
            )
        )

    return columns


# ============================================================
# HEADLINE DETECTION
# ============================================================

def looks_like_headline(block):

    text = block["text"].strip()

    words = text.split()

    if not words:
        return False

    # Native PDF
    font = block.get(
        "max_font",
        block.get(
            "font_height",
            0
        )
    )

    # Short block + larger font
    if len(words) <= 18 and font >= 13:
        return True

    # OCR fallback:
    # short blocks that visually occupy a wider/stronger area
    if len(words) <= 12 and block.get(
        "h",
        0
    ) >= 18:

        return True

    return False


# ============================================================
# ARTICLE START / END DETECTION
# ============================================================

def split_column_into_articles(
    column,
    native_pdf=False
):

    if not column:
        return []

    articles = []

    current = []

    for index, block in enumerate(
        column
    ):

        if not current:

            current = [block]
            continue

        previous = current[-1]

        gap = (
            block["y"]
            -
            previous["bottom"]
        )

        # ----------------------------------------------------
        # HEADLINE AFTER A CLEAR GAP
        # = NEW ARTICLE
        # ----------------------------------------------------

        new_headline = looks_like_headline(
            block
        )

        previous_was_body = (
            len(
                previous["text"].split()
            ) >= 15
        )

        # Typical article separation
        gap_threshold = max(
            12 if native_pdf else 20,
            previous["h"] * (
                0.85 if native_pdf else 1.4
            )
        )

        if (
            new_headline
            and previous_was_body
            and gap >= gap_threshold
        ):

            articles.append(
                current
            )

            current = [block]

        else:

            # ------------------------------------------------
            # VERY LARGE GAP
            # Definitely a new article/section
            # ------------------------------------------------

            large_gap = max(
                35 if native_pdf else 55,
                previous["h"] * 3.0
            )

            if gap > large_gap:

                articles.append(
                    current
                )

                current = [block]

            else:

                current.append(
                    block
                )

    if current:
        articles.append(
            current
        )

    return articles


# ============================================================
# ARTICLE TEXT
# ============================================================

def blocks_to_text(blocks):

    blocks = sorted(
        blocks,
        key=lambda b: (
            b["y"],
            b["x"]
        )
    )

    parts = []

    for block in blocks:

        text = clean(
            block["text"]
        )

        if text:
            parts.append(text)

    return " ".join(parts)


# ============================================================
# ARTICLE QUALITY
# ============================================================

def article_is_valid(text):

    text = clean(text)

    words = text.split()

    if len(words) < 20:
        return False

    letters = sum(
        c.isalpha()
        for c in text
    )

    if len(text) == 0:
        return False

    if (
        letters / len(text)
    ) < 0.55:

        return False

    return True


# ============================================================
# DUPLICATE ARTICLES
# ============================================================

def article_similarity(a, b):

    a_words = set(
        normalize(a).split()
    )

    b_words = set(
        normalize(b).split()
    )

    if not a_words or not b_words:
        return 0

    return len(
        a_words & b_words
    ) / len(
        a_words | b_words
    )


def remove_duplicates(articles):

    final = []

    for article in articles:

        duplicate = False

        for old in final:

            similarity = article_similarity(
                article,
                old
            )

            if similarity >= 0.65:

                duplicate = True
                break

        if not duplicate:
            final.append(
                article
            )

    return final


# ============================================================
# EXTRACT ARTICLES FROM BLOCKS
# ============================================================

def extract_articles_from_blocks(
    blocks,
    page_width,
    native_pdf=False
):

    if not blocks:
        return []

    columns = detect_columns(
        blocks,
        page_width
    )

    all_articles = []

    for column in columns:

        article_groups = split_column_into_articles(
            column,
            native_pdf=native_pdf
        )

        for group in article_groups:

            text = blocks_to_text(
                group
            )

            if article_is_valid(
                text
            ):

                all_articles.append(
                    text
                )

    return all_articles


# ============================================================
# PROCESS NATIVE PDF
# ============================================================

def process_native_page(page):

    blocks = get_pdf_blocks(
        page
    )

    if not blocks:
        return []

    total_words = sum(
        len(
            b["text"].split()
        )
        for b in blocks
    )

    if total_words < 50:
        return []

    articles = extract_articles_from_blocks(
        blocks,
        page.rect.width,
        native_pdf=True
    )

    return articles


# ============================================================
# PROCESS OCR PAGE
# ============================================================

def process_ocr_page(image):

    words = ocr_words(
        image
    )

    if not words:
        return []

    blocks = make_ocr_blocks(
        words
    )

    if not blocks:
        return []

    articles = extract_articles_from_blocks(
        blocks,
        image.width,
        native_pdf=False
    )

    return articles


# ============================================================
# FILTER COMPLETED ARTICLES
# ============================================================

def filter_defence_articles(
    articles
):

    final = []

    for article in articles:

        # IMPORTANT:
        # The entire article is already extracted.
        # Only now classify it.

        if is_defence_article(
            article
        ):

            final.append(
                article
            )

    return remove_duplicates(
        final
    )


# ============================================================
# PDF PAGE PROCESSOR
# ============================================================

def process_pdf_page(page):

    # --------------------------------------------------------
    # 1. TRY NATIVE PDF LAYOUT
    # --------------------------------------------------------

    native_articles = process_native_page(
        page
    )

    native_defence = filter_defence_articles(
        native_articles
    )

    if native_defence:

        return native_defence, "Native PDF"

    # --------------------------------------------------------
    # 2. FALLBACK TO OCR
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
        (
            pix.width,
            pix.height
        ),
        pix.samples
    )

    ocr_articles = process_ocr_page(
        image
    )

    defence_articles = filter_defence_articles(
        ocr_articles
    )

    return defence_articles, "OCR"


# ============================================================
# FILE HASH
# ============================================================

def file_hash(data):

    return hashlib.md5(
        data
    ).hexdigest()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "Layout-aware extraction of complete defence-related newspaper articles."
)


# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(
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
# MAIN
# ============================================================

if uploaded:

    data = uploaded.getvalue()

    current_hash = file_hash(
        data
    )

    if (
        current_hash
        !=
        st.session_state.processed_hash
    ):

        st.session_state.results = []

        st.session_state.processed_hash = (
            current_hash
        )

    filename = uploaded.name

    # ========================================================
    # PDF
    # ========================================================

    if filename.lower().endswith(
        ".pdf"
    ):

        pdf = fitz.open(
            stream=data,
            filetype="pdf"
        )

        total_pages = len(pdf)

        st.info(
            f"📄 {filename} — {total_pages} pages"
        )

        mode = st.radio(
            "Pages to scan",
            [
                "Test first 3 pages",
                "All pages",
                "Select pages"
            ],
            horizontal=True
        )

        if mode == "Test first 3 pages":

            selected = list(
                range(
                    min(
                        3,
                        total_pages
                    )
                )
            )

        elif mode == "All pages":

            selected = list(
                range(
                    total_pages
                )
            )

        else:

            selected_numbers = st.multiselect(
                "Select page numbers",
                list(
                    range(
                        1,
                        total_pages + 1
                    )
                )
            )

            selected = [
                x - 1
                for x in selected_numbers
            ]

        scan = st.button(
            "🚀 SCAN NEWSPAPER",
            type="primary"
        )

        if scan and selected:

            results = []

            progress = st.progress(
                0
            )

            status = st.empty()

            native_pages = 0
            ocr_pages = 0

            for position, page_no in enumerate(
                selected
            ):

                status.info(
                    f"🔎 Processing page "
                    f"{page_no + 1} "
                    f"of {total_pages}"
                )

                page = pdf.load_page(
                    page_no
                )

                try:

                    articles, method = (
                        process_pdf_page(
                            page
                        )
                    )

                    if method == "Native PDF":
                        native_pages += 1
                    else:
                        ocr_pages += 1

                    results.extend(
                        articles
                    )

                except Exception as error:

                    st.warning(
                        f"Page {page_no + 1}: "
                        f"{error}"
                    )

                progress.progress(
                    (position + 1)
                    /
                    len(selected)
                )

            # ------------------------------------------------
            # FINAL DEDUP
            # ------------------------------------------------

            results = remove_duplicates(
                results
            )

            st.session_state.results = (
                results
            )

            status.success(
                "✅ Scan completed."
            )

            st.write(
                f"Pages processed: **{len(selected)}**"
            )

            st.write(
                f"Native PDF pages: **{native_pages}**"
            )

            st.write(
                f"OCR pages: **{ocr_pages}**"
            )

            st.write(
                f"Defence articles extracted: "
                f"**{len(results)}**"
            )

        pdf.close()

    # ========================================================
    # IMAGE
    # ========================================================

    else:

        scan = st.button(
            "🚀 SCAN NEWSPAPER IMAGE",
            type="primary"
        )

        if scan:

            image = Image.open(
                uploaded
            )

            with st.spinner(
                "🔎 Reading newspaper layout..."
            ):

                all_articles = process_ocr_page(
                    image
                )

                results = filter_defence_articles(
                    all_articles
                )

                st.session_state.results = (
                    results
                )

            st.success(
                "✅ Scan completed."
            )

            st.write(
                f"Defence articles extracted: "
                f"**{len(results)}**"
            )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results


if results:

    st.divider()

    st.header(
        "📰 Complete Defence Articles"
    )

    st.success(
        f"{len(results)} defence article(s) extracted."
    )

    for number, article in enumerate(
        results,
        start=1
    ):

        with st.container(
            border=True
        ):

            st.subheader(
                f"News {number}"
            )

            st.write(
                article
            )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    downloadable = []

    for number, article in enumerate(
        results,
        start=1
    ):

        downloadable.append(
            f"NEWS {number}\n"
            f"{'=' * 80}\n"
            f"{article}\n"
        )

    st.download_button(
        "⬇️ Download Extracted Defence News",
        data="\n".join(
            downloadable
        ),
        file_name="defence_news.txt",
        mime="text/plain"
    )

else:

    if uploaded:

        st.warning(
            "No defence articles found in the selected pages."
        )
