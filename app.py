import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output
import fitz
import pandas as pd
import re
import statistics


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
    "The system detects complete defence-related articles "
    "using text extraction, OCR and newspaper layout analysis."
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
    "military operation"
]


DEFENCE_SPECIFIC = [
    "missile",
    "missiles",
    "rocket",
    "rockets",
    "warship",
    "warships",
    "fighter aircraft",
    "fighter jet",
    "fighter jets",
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
    "tanks",
    "armoured",
    "armored",
    "drone",
    "drones",
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
    "commandos",
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
    "strategic forces"
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
    "neutralized"
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
    "television serial"
]


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

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
        1 for x in DEFENCE_STRONG
        if x in t
    )

    specific = sum(
        1 for x in DEFENCE_SPECIFIC
        if x in t
    )

    context = sum(
        1 for x in DEFENCE_CONTEXT
        if x in t
    )

    negative = sum(
        1 for x in NON_DEFENCE
        if x in t
    )

    return (
        strong * 6
        + specific * 3
        + context
        - negative * 4
    )


# =========================================================
# DEFENCE CLASSIFICATION
# =========================================================

def is_defence_article(text):

    t = normalize_text(text)

    if len(t.split()) < 8:
        return False

    strong = any(
        x in t
        for x in DEFENCE_STRONG
    )

    specific = sum(
        1 for x in DEFENCE_SPECIFIC
        if x in t
    )

    context = sum(
        1 for x in DEFENCE_CONTEXT
        if x in t
    )

    negative = sum(
        1 for x in NON_DEFENCE
        if x in t
    )

    if negative >= 2 and not strong:
        return False

    if strong:
        return True

    if specific >= 2 and context >= 1:
        return True

    terror = any(
        x in t
        for x in [
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
            "anti-terror"
        ]
    )

    security = any(
        x in t
        for x in [
            "security forces",
            "armed forces",
            "troops",
            "army personnel",
            "military personnel",
            "operation",
            "captured",
            "neutralised",
            "neutralized"
        ]
    )

    return terror and security


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.replace(
        "- ",
        ""
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# NATIVE PDF TEXT EXTRACTION
#
# MUCH FASTER than OCR when PDF already contains text.
# =========================================================

def extract_pdf_native_page(page):

    blocks = page.get_text(
        "blocks"
    )

    result = []

    for block in blocks:

        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = (
            block[:5]
        )

        text = clean_text(text)

        if len(text) < 15:
            continue

        result.append({
            "text": text,
            "x": float(x0),
            "y": float(y0),
            "right": float(x1),
            "bottom": float(y1)
        })

    return result


# =========================================================
# CHECK WHETHER NATIVE TEXT IS USABLE
# =========================================================

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


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    width, height = image.size

    # Do NOT over-enlarge.
    # This keeps OCR reasonably fast.
    target_width = 2400

    if width < target_width:

        scale = (
            target_width / width
        )

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
    ).enhance(1.5)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# =========================================================
# OCR DATA
# =========================================================

def get_ocr_blocks(image):

    processed = preprocess_image(
        image
    )

    config = "--oem 3 --psm 3"

    data = pytesseract.image_to_data(
        processed,
        output_type=Output.DATAFRAME,
        config=config
    )

    if data.empty:
        return []

    data = data.dropna(
        subset=["text"]
    )

    data["text"] = (
        data["text"]
        .astype(str)
        .str.strip()
    )

    data["conf"] = pd.to_numeric(
        data["conf"],
        errors="coerce"
    )

    data = data[
        (data["text"] != "")
        &
        (data["conf"] >= 25)
    ]

    if data.empty:
        return []

    blocks = []

    # -----------------------------------------------------
    # IMPORTANT:
    # Use Tesseract BLOCK/PARAGRAPH information.
    # Do NOT globally join every line on the page.
    # -----------------------------------------------------

    grouped = data.groupby(
        [
            "block_num",
            "par_num"
        ],
        sort=False
    )

    for _, group in grouped:

        group = group.sort_values(
            [
                "top",
                "left"
            ]
        )

        words = group[
            "text"
        ].tolist()

        text = " ".join(
            words
        )

        text = clean_text(
            text
        )

        if len(text) < 15:
            continue

        blocks.append({

            "text": text,

            "x": float(
                group["left"].min()
            ),

            "y": float(
                group["top"].min()
            ),

            "right": float(
                (
                    group["left"]
                    +
                    group["width"]
                ).max()
            ),

            "bottom": float(
                (
                    group["top"]
                    +
                    group["height"]
                ).max()
            )
        })

    return blocks


# =========================================================
# SPLIT NATIVE BLOCKS INTO COLUMNS
# =========================================================

def group_into_columns(
    blocks,
    page_width
):

    if not blocks:
        return []

    # Sort left-to-right first
    blocks = sorted(
        blocks,
        key=lambda b: (
            b["x"],
            b["y"]
        )
    )

    columns = []

    for block in blocks:

        placed = False

        center = (
            block["x"]
            +
            block["right"]
        ) / 2

        for column in columns:

            column_left = min(
                b["x"]
                for b in column
            )

            column_right = max(
                b["right"]
                for b in column
            )

            column_center = (
                column_left
                +
                column_right
            ) / 2

            tolerance = max(
                35,
                page_width * 0.025
            )

            if abs(
                center
                -
                column_center
            ) <= tolerance:

                column.append(
                    block
                )

                placed = True

                break

        if not placed:

            columns.append(
                [block]
            )

    # Sort each column top-to-bottom
    for column in columns:

        column.sort(
            key=lambda b: (
                b["y"],
                b["x"]
            )
        )

    # Columns left-to-right
    columns.sort(
        key=lambda c: min(
            b["x"]
            for b in c
        )
    )

    return columns


# =========================================================
# JOIN BLOCKS INSIDE A COLUMN
#
# This preserves a complete article but does not mix
# neighbouring newspaper columns.
# =========================================================

def build_article_candidates(
    blocks,
    page_width
):

    if not blocks:
        return []

    columns = group_into_columns(
        blocks,
        page_width
    )

    candidates = []

    for column in columns:

        if not column:
            continue

        current = []

        previous = None

        for block in column:

            if previous is None:

                current = [
                    block
                ]

                previous = block

                continue

            gap = (
                block["y"]
                -
                previous["bottom"]
            )

            previous_height = (
                previous["bottom"]
                -
                previous["y"]
            )

            # Large vertical whitespace
            # usually indicates another article.
            article_break = (
                gap
                >
                max(
                    35,
                    previous_height * 2.8
                )
            )

            if article_break:

                if current:

                    candidates.append(
                        current
                    )

                current = [
                    block
                ]

            else:

                current.append(
                    block
                )

            previous = block

        if current:

            candidates.append(
                current
            )

    results = []

    for group in candidates:

        text_parts = [
            b["text"]
            for b in group
        ]

        text = clean_text(
            " ".join(
                text_parts
            )
        )

        word_count = len(
            text.split()
        )

        # Ignore tiny OCR fragments
        if word_count < 10:
            continue

        results.append({

            "text": text,

            "x": min(
                b["x"]
                for b in group
            ),

            "y": min(
                b["y"]
                for b in group
            ),

            "right": max(
                b["right"]
                for b in group
            ),

            "bottom": max(
                b["bottom"]
                for b in group
            )
        })

    return results


# =========================================================
# SPLIT OVERLY LARGE MIXED CANDIDATE
# =========================================================

def split_large_candidate(
    candidate
):

    text = candidate["text"]

    words = text.split()

    # Very large text blocks can still be multiple articles.
    if len(words) <= 220:

        return [
            candidate
        ]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current = []

    count = 0

    for sentence in sentences:

        current.append(
            sentence
        )

        count += len(
            sentence.split()
        )

        if count >= 100:

            chunks.append(
                " ".join(current)
            )

            current = []

            count = 0

    if current:

        chunks.append(
            " ".join(current)
        )

    results = []

    for chunk in chunks:

        if len(chunk.split()) >= 10:

            new_candidate = dict(
                candidate
            )

            new_candidate[
                "text"
            ] = clean_text(chunk)

            results.append(
                new_candidate
            )

    return results


# =========================================================
# FILTER DEFENCE ARTICLES
# =========================================================

def filter_defence_articles(
    candidates
):

    results = []

    for candidate in candidates:

        pieces = split_large_candidate(
            candidate
        )

        for piece in pieces:

            text = piece["text"]

            score = defence_score(
                text
            )

            if is_defence_article(
                text
            ):

                piece["score"] = score

                results.append(
                    piece
                )

    return results


# =========================================================
# SIMILARITY
# =========================================================

def similarity(a, b):

    a_words = set(
        normalize_text(a).split()
    )

    b_words = set(
        normalize_text(b).split()
    )

    if not a_words or not b_words:
        return 0

    return (
        len(
            a_words & b_words
        )
        /
        len(
            a_words | b_words
        )
    )


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicate_articles(
    candidates
):

    final = []

    # Longer articles first
    candidates = sorted(
        candidates,
        key=lambda x: len(
            x["text"]
        ),
        reverse=True
    )

    for candidate in candidates:

        text = candidate[
            "text"
        ]

        duplicate = False

        for existing in final:

            sim = similarity(
                text,
                existing["text"]
            )

            if sim >= 0.55:

                duplicate = True

                break

        if not duplicate:

            final.append(
                candidate
            )

    # Restore page reading order
    final.sort(
        key=lambda x: (
            x["y"],
            x["x"]
        )
    )

    return final


# =========================================================
# PROCESS IMAGE PAGE
# =========================================================

def process_image_page(
    image
):

    blocks = get_ocr_blocks(
        image
    )

    if not blocks:

        return [], ""

    width = image.width

    candidates = build_article_candidates(
        blocks,
        width
    )

    defence = filter_defence_articles(
        candidates
    )

    final = remove_duplicate_articles(
        defence
    )

    complete_ocr = "\n\n".join(
        b["text"]
        for b in blocks
    )

    return (
        final,
        complete_ocr
    )


# =========================================================
# PROCESS PDF PAGE
# =========================================================

def process_pdf_page(
    page
):

    # -----------------------------------------------------
    # FIRST TRY NATIVE PDF TEXT
    # -----------------------------------------------------

    native_blocks = (
        extract_pdf_native_page(
            page
        )
    )

    if native_text_is_usable(
        native_blocks
    ):

        candidates = (
            build_article_candidates(
                native_blocks,
                page.rect.width
            )
        )

        defence = (
            filter_defence_articles(
                candidates
            )
        )

        final = (
            remove_duplicate_articles(
                defence
            )
        )

        complete_ocr = "\n\n".join(
            b["text"]
            for b in native_blocks
        )

        return (
            final,
            complete_ocr,
            "Native PDF text"
        )


    # -----------------------------------------------------
    # SCANNED PDF → OCR
    # -----------------------------------------------------

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

    final, text = (
        process_image_page(
            image
        )
    )

    del image
    del pix

    return (
        final,
        text,
        "Tesseract OCR"
    )


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:

    st.session_state.results = None


# =========================================================
# FILE UPLOAD
# =========================================================

file = st.file_uploader(
    "📂 Upload Newspaper",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ]
)


# =========================================================
# UPLOADED FILE
# =========================================================

if file:

    st.success(
        f"✅ Uploaded: {file.name}"
    )

    total_pages = 1

    # -----------------------------------------------------
    # PDF INFO
    # -----------------------------------------------------

    if file.type == "application/pdf":

        pdf_bytes = file.getvalue()

        temp_doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total_pages = len(
            temp_doc
        )

        temp_doc.close()

        st.info(
            f"📑 PDF contains "
            f"{total_pages} pages."
        )


    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    elif file.type.startswith(
        "image"
    ):

        preview = Image.open(
            file
        )

        st.image(
            preview,
            caption="Uploaded Newspaper",
            use_container_width=True
        )


    # -----------------------------------------------------
    # PAGE SELECTION
    # -----------------------------------------------------

    if total_pages > 1:

        max_pages = st.number_input(
            "📄 Pages to scan",
            min_value=1,
            max_value=total_pages,
            value=total_pages,
            step=1
        )

    else:

        max_pages = 1


    # -----------------------------------------------------
    # SCAN BUTTON
    # -----------------------------------------------------

    scan = st.button(
        "🚀 Scan Newspaper",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # SCANNING
    # =====================================================

    if scan:

        all_articles = []

        all_ocr = []

        pages_scanned = 0

        native_pages = 0

        ocr_pages = 0

        progress = st.progress(
            0
        )

        status = st.empty()


        # =================================================
        # IMAGE
        # =================================================

        if file.type.startswith(
            "image"
        ):

            status.info(
                "🔍 Scanning newspaper image..."
            )

            image = Image.open(
                file
            ).convert("RGB")

            articles, text = (
                process_image_page(
                    image
                )
            )

            all_articles.extend(
                articles
            )

            all_ocr.append(
                "--- IMAGE ---\n"
                + text
            )

            pages_scanned = 1

            ocr_pages = 1

            del image

            progress.progress(
                1.0
            )


        # =================================================
        # PDF
        # =================================================

        else:

            pdf_bytes = file.getvalue()

            doc = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

            selected = int(
                max_pages
            )


            # ---------------------------------------------
            # EXACTLY ONE PASS
            # ---------------------------------------------

            for index in range(
                selected
            ):

                page_number = (
                    index + 1
                )

                status.info(
                    f"🔍 Analysing page "
                    f"{page_number} / "
                    f"{selected}"
                )

                page = doc.load_page(
                    index
                )

                (
                    articles,
                    text,
                    method
                ) = process_pdf_page(
                    page
                )

                all_articles.extend(
                    articles
                )

                all_ocr.append(
                    f"\n--- PAGE "
                    f"{page_number} "
                    f"({method}) ---\n"
                    f"{text}"
                )

                pages_scanned = (
                    page_number
                )

                if method == "Native PDF text":

                    native_pages += 1

                else:

                    ocr_pages += 1


                progress.progress(
                    page_number
                    /
                    selected
                )


            doc.close()


        # =================================================
        # FINAL GLOBAL DEDUPLICATION
        # =================================================

        status.info(
            "🧠 Combining complete "
            "defence articles..."
        )

        final_articles = (
            remove_duplicate_articles(
                all_articles
            )
        )


        # -------------------------------------------------
        # Save result
        # -------------------------------------------------

        st.session_state.results = {

            "articles": final_articles,

            "ocr": "\n".join(
                all_ocr
            ),

            "pages": pages_scanned,

            "native": native_pages,

            "ocr_pages": ocr_pages,

            "filename": file.name
        }


        status.success(
            "✅ Analysis completed."
        )


        progress.progress(
            1.0
        )


# =========================================================
# RESULTS
# =========================================================

if st.session_state.results:

    result = (
        st.session_state.results
    )

    articles = result[
        "articles"
    ]

    st.divider()

    st.header(
        "📰 Defence News Extracted"
    )


    if articles:

        st.success(
            f"🛡️ {len(articles)} "
            f"complete defence-related "
            f"articles detected."
        )


        for i, article in enumerate(
            articles,
            start=1
        ):

            st.markdown(
                f"### 📰 News {i}"
            )

            st.write(
                article["text"]
            )

            st.caption(
                f"Defence relevance score: "
                f"{article.get('score', 0)}"
            )

            st.divider()


        # =================================================
        # DOWNLOAD
        # =================================================

        download_text = "\n\n".join(

            [
                (
                    f"NEWS {i}\n\n"
                    f"{article['text']}"
                )

                for i, article in enumerate(
                    articles,
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
            "⚠️ No defence-related "
            "articles detected."
        )


    # =====================================================
    # OCR DEBUG
    # =====================================================

    with st.expander(
        "🔎 View Complete Extracted Text"
    ):

        st.text_area(
            "Complete Text",
            result["ocr"],
            height=500
        )


    # =====================================================
    # STATUS
    # =====================================================

    st.divider()

    st.subheader(
        "📊 Scanner Status"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Pages Scanned",
        result["pages"]
    )

    col2.metric(
        "Defence Articles",
        len(articles)
    )

    col3.metric(
        "Native PDF Pages",
        result["native"]
    )

    col4.metric(
        "OCR Pages",
        result["ocr_pages"]
    )
