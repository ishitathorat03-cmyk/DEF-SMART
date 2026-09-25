import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output
import fitz
import pandas as pd
import re
import statistics


# =========================================================
# TESSERACT CONFIGURATION
# =========================================================

# If Tesseract is available in PATH:
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# If Windows cannot find Tesseract, uncomment and use:
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )


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
    "The system detects complete defence-related news articles "
    "using OCR, newspaper layout and defence-context analysis."
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
    "military strike"
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
# TEXT NORMALIZATION
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
        1
        for word in DEFENCE_STRONG
        if word in t
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

    score = (
        strong * 6
        + specific * 3
        + context
        - negative * 4
    )

    return score


# =========================================================
# DEFENCE ARTICLE CLASSIFIER
# =========================================================

def is_defence_article(text):

    t = normalize_text(text)

    if not t:
        return False

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

    # Strong negative article
    if negative >= 2 and not strong:
        return False

    # Strong defence evidence
    if strong:
        return True

    # Specific military evidence
    if specific >= 2 and context >= 1:
        return True

    # Terrorism / security combination
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
            "anti-terror"
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
            "neutralised",
            "neutralized"
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

    # Upscale small newspaper images
    target_width = 3000

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
    ).enhance(1.7)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# =========================================================
# OCR
# =========================================================

def get_ocr_data(image):

    processed = preprocess_image(
        image
    )

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

    # Convert confidence safely
    data["conf"] = pd.to_numeric(
        data["conf"],
        errors="coerce"
    )

    data = data[
        data["conf"] >= 20
    ]

    return data, processed


# =========================================================
# BUILD OCR LINES
# =========================================================

def build_lines(data):

    if data.empty:
        return []

    words = []

    for _, row in data.iterrows():

        words.append({

            "text": str(row["text"]),

            "x": int(row["left"]),
            "y": int(row["top"]),

            "w": int(row["width"]),
            "h": int(row["height"]),

            "block": int(row["block_num"]),
            "paragraph": int(row["par_num"]),
            "line": int(row["line_num"])
        })


    # Sort top → bottom → left → right
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
            word["y"]
            +
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
                word["h"] * 0.75
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

            "right": max(
                w["x"] + w["w"]
                for w in line
            ),

            "bottom": max(
                w["y"] + w["h"]
                for w in line
            ),

            "height": max(
                w["h"]
                for w in line
            )
        })


    result.sort(
        key=lambda x: (
            x["y"],
            x["x"]
        )
    )


    return result


# =========================================================
# ESTIMATE LINE HEIGHT
# =========================================================

def estimate_line_height(lines):

    if not lines:
        return 20

    heights = [
        line["height"]
        for line in lines
        if line["height"] > 0
    ]

    if not heights:
        return 20

    return max(
        12,
        statistics.median(heights)
    )


# =========================================================
# ARTICLE SEGMENTATION
#
# IMPORTANT:
# We DO NOT treat every line as an article.
# We build groups of continuous newspaper text.
# =========================================================

def segment_articles(lines):

    if not lines:
        return []


    base_height = estimate_line_height(
        lines
    )


    # -----------------------------------------------------
    # First build spatial groups.
    # Lines close vertically and horizontally are grouped.
    # -----------------------------------------------------

    groups = []

    current = []


    for i, line in enumerate(lines):

        current.append(line)


        if i == len(lines) - 1:

            break


        next_line = lines[i + 1]


        vertical_gap = (
            next_line["y"]
            -
            line["bottom"]
        )


        # Horizontal relationship
        horizontal_gap = (
            next_line["x"]
            -
            line["right"]
        )


        # Normal continuation
        normal_continuation = (
            vertical_gap
            <=
            base_height * 1.8
        )


        # Large whitespace indicates article/column break
        large_break = (
            vertical_gap
            >
            base_height * 3.0
        )


        if large_break:

            groups.append(
                current
            )

            current = []


    if current:

        groups.append(
            current
        )


    # -----------------------------------------------------
    # Convert groups into text
    # -----------------------------------------------------

    candidates = []


    for group in groups:

        text = " ".join(
            line["text"]
            for line in group
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()


        if len(text) < 30:

            continue


        candidates.append({

            "text": text,

            "x": min(
                line["x"]
                for line in group
            ),

            "y": min(
                line["y"]
                for line in group
            ),

            "right": max(
                line["right"]
                for line in group
            ),

            "bottom": max(
                line["bottom"]
                for line in group
            )
        })


    # -----------------------------------------------------
    # Defence filtering
    # -----------------------------------------------------

    defence_candidates = []


    for candidate in candidates:

        text = candidate["text"]

        score = defence_score(
            text
        )


        if is_defence_article(
            text
        ):

            candidate["score"] = score

            defence_candidates.append(
                candidate
            )


    return [
        candidate["text"]
        for candidate in defence_candidates
    ]


# =========================================================
# MERGE ARTICLES
#
# Prevents one article from becoming multiple news items.
# =========================================================

def merge_articles(articles):

    if not articles:
        return []


    cleaned = []


    for article in articles:

        article = re.sub(
            r"\s+",
            " ",
            article
        ).strip()


        if len(article) < 30:

            continue


        cleaned.append(
            article
        )


    final = []


    for article in cleaned:

        article_normalized = normalize_text(
            article
        )

        article_words = set(
            article_normalized.split()
        )


        if not article_words:

            continue


        merged = False


        for i, existing in enumerate(
            final
        ):

            existing_normalized = (
                normalize_text(existing)
            )

            existing_words = set(
                existing_normalized.split()
            )


            if not existing_words:

                continue


            intersection = (
                article_words
                &
                existing_words
            )


            union = (
                article_words
                |
                existing_words
            )


            if not union:

                continue


            overlap = (
                len(intersection)
                /
                len(union)
            )


            # Same article / overlapping OCR
            if overlap >= 0.30:

                if len(article) > len(
                    existing
                ):

                    final[i] = article

                merged = True

                break


        if not merged:

            final.append(
                article
            )


    return final


# =========================================================
# REMOVE VERY SIMILAR ARTICLES
# =========================================================

def remove_similar_articles(
    articles
):

    final = []


    for article in articles:

        normalized = normalize_text(
            article
        )


        duplicate = False


        for existing in final:

            existing_normalized = (
                normalize_text(existing)
            )


            # Exact-ish duplicate
            if (
                normalized
                in
                existing_normalized
            ):

                duplicate = True

                break


            if (
                existing_normalized
                in
                normalized
            ):

                duplicate = True

                break


        if not duplicate:

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


    articles = remove_similar_articles(
        articles
    )


    complete_text = "\n".join(
        line["text"]
        for line in lines
    )


    return (
        articles,
        complete_text
    )


# =========================================================
# SESSION STATE
# =========================================================

if "results" not in st.session_state:

    st.session_state.results = None


# =========================================================
# FILE UPLOADER
# =========================================================

file = st.file_uploader(
    "📂 Upload Newspaper Image or PDF",
    type=[
        "png",
        "jpg",
        "jpeg",
        "pdf"
    ]
)


# =========================================================
# MAIN UI
# =========================================================

if file:

    st.success(
        f"✅ File uploaded: {file.name}"
    )


    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    if file.type.startswith(
        "image"
    ):

        image_preview = Image.open(
            file
        )

        st.image(
            image_preview,
            caption="Uploaded Newspaper",
            use_container_width=True
        )


    # -----------------------------------------------------
    # PDF INFORMATION
    # -----------------------------------------------------

    total_pages = 1

    if file.type == "application/pdf":

        pdf_bytes_preview = file.getvalue()

        preview_doc = fitz.open(
            stream=pdf_bytes_preview,
            filetype="pdf"
        )

        total_pages = len(
            preview_doc
        )

        preview_doc.close()


        st.info(
            f"📑 PDF contains "
            f"{total_pages} pages."
        )


    # -----------------------------------------------------
    # PAGE LIMIT
    # -----------------------------------------------------

    if total_pages > 1:

        max_pages = st.number_input(
            "📄 Number of pages to scan",
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

    scan_clicked = st.button(
        "🚀 Scan Newspaper",
        type="primary",
        use_container_width=True
    )


    # =====================================================
    # PROCESS ONLY WHEN BUTTON IS CLICKED
    # =====================================================

    if scan_clicked:

        all_articles = []

        all_ocr = []

        pages_scanned = 0


        progress = st.progress(
            0
        )


        status = st.empty()


        # =================================================
        # IMAGE PROCESSING
        # =================================================

        if file.type.startswith(
            "image"
        ):

            status.info(
                "🔍 Analysing newspaper image..."
            )


            image = Image.open(
                file
            )


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


            progress.progress(
                1.0
            )


        # =================================================
        # PDF PROCESSING
        # =================================================

        elif file.type == "application/pdf":

            pdf_bytes = file.getvalue()


            doc = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )


            selected_pages = int(
                max_pages
            )


            # ---------------------------------------------
            # IMPORTANT:
            # ONE PDF OPEN
            # ONE LOOP
            # ONE OCR PER PAGE
            # ---------------------------------------------

            for page_index in range(
                selected_pages
            ):

                page_number = (
                    page_index + 1
                )


                status.info(
                    f"🔍 Analysing page "
                    f"{page_number} of "
                    f"{selected_pages}..."
                )


                page = doc.load_page(
                    page_index
                )


                # High resolution rendering
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        2.5,
                        2.5
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


                articles, ocr_text = (
                    process_image(image)
                )


                # Add page results
                all_articles.extend(
                    articles
                )


                all_ocr.append(
                    f"\n--- PAGE "
                    f"{page_number} ---\n"
                    f"{ocr_text}"
                )


                pages_scanned = (
                    page_number
                )


                progress.progress(
                    page_number
                    /
                    selected_pages
                )


            doc.close()


        # =================================================
        # FINAL ARTICLE PROCESSING
        # =================================================

        status.info(
            "🧠 Combining and cleaning "
            "detected defence articles..."
        )


        final_articles = merge_articles(
            all_articles
        )


        final_articles = (
            remove_similar_articles(
                final_articles
            )
        )


        # Save results
        st.session_state.results = {

            "articles": final_articles,

            "ocr": "\n".join(
                all_ocr
            ),

            "pages": pages_scanned,

            "filename": file.name
        }


        status.success(
            "✅ Newspaper analysis completed."
        )


        progress.progress(
            1.0
        )


# =========================================================
# SHOW RESULTS
# =========================================================

if st.session_state.results:

    results = (
        st.session_state.results
    )


    final_articles = (
        results["articles"]
    )


    all_ocr = (
        results["ocr"]
    )


    pages_scanned = (
        results["pages"]
    )


    st.divider()


    st.header(
        "📰 Defence News Extracted"
    )


    if final_articles:

        st.success(
            f"🛡️ "
            f"{len(final_articles)} "
            f"defence-related news articles found"
        )


        # -------------------------------------------------
        # ARTICLE DISPLAY
        # -------------------------------------------------

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


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        download_text = "\n\n".join(

            [
                (
                    f"NEWS {i}\n\n"
                    f"{article}"
                )

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
            "was detected in the uploaded newspaper."
        )


    # =====================================================
    # OCR DEBUG
    # =====================================================

    with st.expander(
        "🔎 View Complete OCR Text"
    ):

        st.text_area(
            "Complete OCR",
            all_ocr,
            height=500
        )


    # =====================================================
    # STATUS
    # =====================================================

    st.divider()


    st.subheader(
        "📊 Scanner Status"
    )


    col1, col2, col3 = st.columns(
        3
    )


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
