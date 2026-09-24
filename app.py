import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output
import fitz
import pandas as pd
import re


# =========================================================
# TESSERACT
# =========================================================

pytesseract.pytesseract.tesseract_cmd = "tesseract"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ Defence News Scanner OCR")

st.write(
    "Upload a newspaper image or PDF. "
    "The system analyses the newspaper layout and extracts "
    "complete defence-related articles."
)


# =========================================================
# DEFENCE VOCABULARY
# =========================================================

DEFENCE_STRONG = [
    "indian army",
    "indian navy",
    "indian air force",
    "air force",
    "iaf",
    "indian armed forces",
    "armed forces",
    "ministry of defence",
    "defence ministry",
    "defense ministry",
    "defence forces",
    "defense forces",
    "military operation",
    "military exercise",
    "military deployment",
    "military training",
    "naval exercise",
    "army exercise",
    "air force exercise",
    "defence deal",
    "defense deal",
    "defence procurement",
    "defense procurement",
    "defence equipment",
    "defense equipment",
    "counter terrorism",
    "counter-terrorism",
    "anti-terror operation",
    "anti terror operation",
    "terrorist attack",
    "terrorist group",
    "terrorist organisation",
    "terrorist organization",
    "militant group",
    "military strike",
]


DEFENCE_SPECIFIC = [
    "missile",
    "rocket",
    "rockets",
    "warship",
    "fighter aircraft",
    "fighter jet",
    "military aircraft",
    "aircraft carrier",
    "submarine",
    "frigate",
    "destroyer",
    "military helicopter",
    "army regiment",
    "regiment",
    "battalion",
    "military base",
    "air base",
    "naval base",
    "military command",
    "defence command",
    "defense command",
    "defence technology",
    "defense technology",
    "weapons system",
    "weapon system",
    "ammunition",
    "artillery",
    "tank",
    "armoured",
    "armored",
    "drone",
    "uav",
    "unmanned aerial vehicle",
    "border security force",
    "border forces",
    "bsf",
    "crpf",
    "itbp",
    "cisf",
    "coast guard",
    "paramilitary",
    "special forces",
    "commando",
    "troops",
    "soldiers",
    "military personnel",
    "militants",
    "terrorists",
    "terrorist",
    "terrorism",
    "terror attack",
    "counterterrorism",
    "insurgency",
    "insurgent",
    "cross border",
    "cross-border",
    "ceasefire",
    "infiltration",
    "infiltrators",
    "line of control",
    "loc",
    "line of actual control",
    "lac",
    "military intelligence",
    "strategic forces",
]


DEFENCE_CONTEXT = [
    "operation",
    "exercise",
    "deployment",
    "training",
    "procurement",
    "security forces",
    "security personnel",
    "border",
    "border security",
    "troops",
    "forces",
    "command",
    "regiment",
    "battalion",
    "aircraft",
    "warship",
    "missile",
    "military",
    "defence",
    "defense",
    "soldiers",
    "terrorist",
    "terrorists",
    "terrorism",
    "militant",
    "militants",
    "counter terror",
    "counter-terror",
    "anti-terror",
    "infiltration",
    "army personnel",
    "naval personnel",
    "air force personnel",
    "captured",
    "neutralised",
    "neutralized",
]


NON_DEFENCE = [
    "school admission",
    "college admission",
    "exam result",
    "stock market",
    "share market",
    "real estate",
    "movie review",
    "film review",
    "celebrity",
    "fashion show",
    "cricket match",
    "football match",
    "recipe",
    "restaurant",
    "wedding",
    "horoscope",
    "property prices",
    "job fair",
    "shopping",
    "television serial",
]


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# DEFENCE SCORE
# =========================================================

def defence_score(text):

    t = normalize_text(text)

    strong = sum(
        1 for word in DEFENCE_STRONG
        if word in t
    )

    specific = sum(
        1 for word in DEFENCE_SPECIFIC
        if word in t
    )

    context = sum(
        1 for word in DEFENCE_CONTEXT
        if word in t
    )

    negative = sum(
        1 for word in NON_DEFENCE
        if word in t
    )

    return (
        strong * 6
        + specific * 3
        + context
        - negative * 4
    )


# =========================================================
# DEFENCE ARTICLE CLASSIFIER
# =========================================================

def is_defence_article(text):

    t = normalize_text(text)

    strong = any(
        word in t
        for word in DEFENCE_STRONG
    )

    specific = sum(
        1
        for word in DEFENCE_SPECIFIC
        if word in t
    )

    context = sum(
        1
        for word in DEFENCE_CONTEXT
        if word in t
    )

    negative = sum(
        1
        for word in NON_DEFENCE
        if word in t
    )

    if negative >= 2 and not strong:
        return False

    if strong:
        return True

    if specific >= 2 and context >= 1:
        return True

    # Counter-terrorism / terrorism
    terror = any(
        word in t
        for word in [
            "terrorist",
            "terrorists",
            "terrorism",
            "terror attack",
            "militant",
            "militants",
            "insurgent",
            "insurgency",
            "counter terror",
            "counter-terror",
            "anti-terror",
        ]
    )

    security = any(
        word in t
        for word in [
            "security forces",
            "armed forces",
            "troops",
            "army personnel",
            "military personnel",
            "operation",
            "captured",
            "arrested",
            "neutralised",
            "neutralized",
        ]
    )

    if terror and security:
        return True

    return False


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    # Increase resolution for small newspaper print
    target_width = 2800

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

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(1.8)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# =========================================================
# LAYOUT-AWARE OCR
# =========================================================

def get_ocr_data(image):

    processed = preprocess_image(image)

    config = "--oem 3 --psm 3"

    data = pytesseract.image_to_data(
        processed,
        output_type=Output.DATAFRAME,
        config=config
    )

    data = data.dropna(
        subset=["text"]
    )

    data["text"] = (
        data["text"]
        .astype(str)
        .str.strip()
    )

    data = data[
        data["text"] != ""
    ]

    # Remove very low-confidence OCR
    data = data[
        data["conf"] >= 20
    ]

    return data, processed


# =========================================================
# RECONSTRUCT LINES FROM OCR COORDINATES
# =========================================================

def build_lines(data):

    words = []

    for _, row in data.iterrows():

        words.append({
            "text": row["text"],
            "x": int(row["left"]),
            "y": int(row["top"]),
            "w": int(row["width"]),
            "h": int(row["height"]),
            "block": int(row["block_num"]),
        })

    # Sort approximately top-to-bottom
    words.sort(
        key=lambda x: (
            x["y"],
            x["x"]
        )
    )

    lines = []

    for word in words:

        placed = False

        word_center = (
            word["y"] +
            word["h"] / 2
        )

        for line in lines:

            line_center = (
                sum(
                    w["y"] + w["h"] / 2
                    for w in line
                )
                /
                len(line)
            )

            tolerance = max(
                12,
                word["h"] * 0.7
            )

            if abs(
                word_center - line_center
            ) <= tolerance:

                line.append(word)

                placed = True
                break

        if not placed:

            lines.append(
                [word]
            )

    # Sort words horizontally
    result = []

    for line in lines:

        line.sort(
            key=lambda x: x["x"]
        )

        text = " ".join(
            w["text"]
            for w in line
        )

        result.append({
            "text": text,
            "x": min(
                w["x"]
                for w in line
            ),
            "y": min(
                w["y"]
                for w in line
            ),
            "bottom": max(
                w["y"] + w["h"]
                for w in line
            ),
        })

    result.sort(
        key=lambda x: (
            x["y"],
            x["x"]
        )
    )

    return result


# =========================================================
# ARTICLE SEGMENTATION
# =========================================================

def segment_articles(lines):

    if not lines:
        return []

    articles = []

    current = []

    for i, line in enumerate(lines):

        current.append(line)

        # Current text
        current_text = " ".join(
            x["text"]
            for x in current
        )

        # -------------------------------------------------
        # Check vertical gap to next line
        # -------------------------------------------------

        if i < len(lines) - 1:

            next_line = lines[i + 1]

            gap = (
                next_line["y"]
                - line["bottom"]
            )

        else:

            gap = 999


        # -------------------------------------------------
        # Large vertical gap often means a new article
        # -------------------------------------------------

        heights = [
            x["bottom"] - x["y"]
            for x in current
        ]

        median_height = (
            sorted(heights)[
                len(heights) // 2
            ]
            if heights
            else 20
        )

        article_break = (
            gap > median_height * 2.5
        )


        # -------------------------------------------------
        # Defence evidence
        # -------------------------------------------------

        defence_now = is_defence_article(
            current_text
        )


        # -------------------------------------------------
        # Close article only when there is a real break
        # -------------------------------------------------

        if article_break:

            if defence_now:

                articles.append(
                    current_text
                )

            current = []


    # Last article
    if current:

        current_text = " ".join(
            x["text"]
            for x in current
        )

        if is_defence_article(
            current_text
        ):

            articles.append(
                current_text
            )


    return articles


# =========================================================
# MERGE OVERLAPPING / SAME ARTICLES
# =========================================================

def merge_articles(articles):

    final = []

    for article in articles:

        article = re.sub(
            r"\s+",
            " ",
            article
        ).strip()

        if len(article) < 20:
            continue

        article_words = set(
            normalize_text(
                article
            ).split()
        )

        merged = False

        for i, existing in enumerate(final):

            existing_words = set(
                normalize_text(
                    existing
                ).split()
            )

            if not article_words or not existing_words:
                continue

            overlap = (
                len(
                    article_words.intersection(
                        existing_words
                    )
                )
                /
                len(
                    article_words.union(
                        existing_words
                    )
                )
            )

            if overlap >= 0.35:

                final[i] = (
                    existing +
                    " " +
                    article
                )

                merged = True
                break

        if not merged:

            final.append(
                article
            )

    return final


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(image):

    data, processed = get_ocr_data(
        image
    )

    lines = build_lines(
        data
    )

    articles = segment_articles(
        lines
    )

    articles = merge_articles(
        articles
    )

    # Complete OCR for debugging
    complete_text = "\n".join(
        line["text"]
        for line in lines
    )

    return (
        articles,
        complete_text
    )


# =========================================================
# FILE UPLOADER
# =========================================================

file = st.file_uploader(
    "📂 Upload Newspaper",
    type=[
        "png",
        "jpg",
        "jpeg",
        "pdf"
    ]
)


# =========================================================
# MAIN PROCESSING
# =========================================================

if file:

    st.success(
        "✅ File uploaded: " +
        file.name
    )

    all_articles = []
    all_ocr = []

    pages_scanned = 0

    # =====================================================
    # IMAGE
    # =====================================================

    if file.type.startswith("image"):

        image = Image.open(file)

        st.image(
            image,
            caption="Uploaded Newspaper",
            use_container_width=True
        )

        with st.spinner(
            "🔍 Analysing newspaper layout..."
        ):

            articles, ocr_text = (
                process_image(image)
            )

        all_articles.extend(
            articles
        )

        all_ocr.append(
            ocr_text
        )

        pages_scanned = 1


    # =====================================================
    # PDF
    # =====================================================

    elif file.type == "application/pdf":

        pdf_bytes = file.read()

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total_pages = len(doc)

        st.info(
            f"📑 PDF contains "
            f"{total_pages} pages."
        )

        max_pages = st.number_input(
            "📄 Pages to scan",
            min_value=1,
            max_value=total_pages,
            value=total_pages,
            step=1
        )

        progress = st.progress(0)

        for page_number, page in enumerate(
            doc,
            start=1
        ):

            if page_number > int(
                max_pages
            ):
                break

            st.write(
                f"🔍 Analysing page "
                f"{page_number} of "
                f"{int(max_pages)}..."
            )

            # High resolution for small articles
            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    2.5,
                    2.5
                )
            )

            image = Image.frombytes(
                "RGB",
                [
                    pix.width,
                    pix.height
                ],
                pix.samples
            )

            articles, ocr_text = (
                process_image(image)
            )

            all_articles.extend(
                articles
            )

            all_ocr.append(
                f"\n--- PAGE {page_number} ---\n"
                f"{ocr_text}"
            )

            pages_scanned = page_number

            progress.progress(
                page_number /
                int(max_pages)
            )


# =========================================================
# FINAL RESULTS
# =========================================================

if file:

    st.divider()

    st.header(
        "📰 Defence News Extracted"
    )

    # Final duplicate removal
    final_articles = merge_articles(
        all_articles
    )

    if final_articles:

        st.success(
            f"🛡️ {len(final_articles)} "
            f"defence-related news articles found"
        )

        for number, article in enumerate(
            final_articles,
            start=1
        ):

            st.markdown(
                f"### 📰 News {number}"
            )

            st.write(
                article
            )

            st.divider()


        # =================================================
        # DOWNLOAD
        # =================================================

        download_text = "\n\n".join(
            [
                f"NEWS {i}\n{article}"
                for i, article in enumerate(
                    final_articles,
                    start=1
                )
            ]
        )

        st.download_button(
            "⬇️ Download Defence News",
            download_text,
            file_name=(
                "defence_news_extracted.txt"
            ),
            mime="text/plain"
        )

    else:

        st.warning(
            "⚠️ No defence-related news "
            "was detected."
        )


    # =====================================================
    # OCR DEBUG
    # =====================================================

    with st.expander(
        "🔎 View Complete OCR Text"
    ):

        st.text_area(
            "OCR Text",
            "\n".join(all_ocr),
            height=500
        )


    # =====================================================
    # STATUS
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Scanner Status"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "OCR Engine",
        "Tesseract"
    )

    col2.metric(
        "Pages Scanned",
        pages_scanned
    )

    col3.metric(
        "Defence Articles",
        len(final_articles)
    )
