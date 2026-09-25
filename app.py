import streamlit as st
import fitz
import re
import hashlib
from collections import defaultdict
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
# TESSERACT
# ============================================================

try:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
except Exception:
    pass


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


def normalize(text):

    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DEFENCE KEYWORDS
# ============================================================

DEFENCE_TERMS = [

    # Indian defence
    "indian army",
    "indian navy",
    "indian air force",
    "armed forces",
    "ministry of defence",
    "ministry of defense",
    "defence ministry",
    "defense ministry",
    "defence minister",
    "defense minister",
    "chief of defence staff",
    "chief of defense staff",
    "chief of army staff",
    "chief of naval staff",
    "chief of air staff",

    # Military
    "military",
    "military operation",
    "military operations",
    "military exercise",
    "military exercises",
    "military deployment",
    "military strike",
    "military strikes",
    "military personnel",
    "military aircraft",
    "military equipment",
    "military base",
    "military training",

    # Services
    "army",
    "navy",
    "air force",
    "army exercise",
    "naval exercise",
    "air force exercise",
    "troops",
    "soldier",
    "soldiers",
    "regiment",
    "regiments",
    "battalion",
    "battalions",
    "brigade",
    "brigades",
    "special forces",
    "commando",
    "commandos",

    # Weapons
    "missile",
    "missiles",
    "ballistic missile",
    "cruise missile",
    "air defence",
    "air defense",
    "air-defence",
    "air-defense",
    "defence missiles",
    "defense missiles",
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
    "combat helicopter",
    "artillery",
    "tank",
    "tanks",
    "armoured vehicle",
    "armored vehicle",
    "ammunition",
    "weapons system",
    "weapon system",
    "radar",
    "drone",
    "drones",
    "uav",

    # Military action
    "air strike",
    "airstrike",
    "airstrikes",
    "missile strike",
    "missile strikes",
    "drone strike",
    "drone strikes",
    "combat",
    "warfare",
    "armed conflict",
    "military conflict",
    "invasion",

    # Security
    "border security",
    "border patrol",
    "border guards",
    "bsf",
    "crpf",
    "itbp",
    "cisf",
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

    # Defence procurement
    "defence procurement",
    "defense procurement",
    "defence deal",
    "defense deal",
    "defence contract",
    "defense contract",

    # International defence
    "ukraine",
    "russia",
    "zelenskyy",
    "zelensky",
    "nato",
    "iran",
    "israel",
    "gaza",
    "pentagon",
    "kremlin",
    "lavrov",
]


# ============================================================
# DEFENCE DETECTION
# ============================================================

def is_defence_article(text):

    text = normalize(text)

    if len(text.split()) < 10:
        return False

    matched = 0

    for term in DEFENCE_TERMS:

        term_normalized = normalize(term)

        if term_normalized in text:

            matched += 1

    return matched >= 1


# ============================================================
# EXTRACT NATIVE PDF BLOCKS
# ============================================================

def extract_pdf_blocks(page):

    blocks = page.get_text(
        "blocks",
        sort=True
    )

    candidates = []

    for block in blocks:

        if len(block) < 7:
            continue

        x0, y0, x1, y1, text, block_no, block_type = block[:7]

        # Only text blocks
        if block_type != 0:
            continue

        text = clean_text(text)

        if not text:
            continue

        if len(text.split()) < 3:
            continue

        candidates.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "text": text
        })

    return candidates


# ============================================================
# GROUP PDF BLOCKS
# ============================================================

def group_pdf_blocks(blocks):

    if not blocks:
        return []

    articles = []

    current = []

    current_x0 = None
    current_x1 = None
    previous_bottom = None

    for block in blocks:

        if not current:

            current = [block]

            current_x0 = block["x0"]
            current_x1 = block["x1"]

            previous_bottom = block["y1"]

            continue

        vertical_gap = (
            block["y0"]
            -
            previous_bottom
        )

        overlap = (
            min(
                current_x1,
                block["x1"]
            )
            -
            max(
                current_x0,
                block["x0"]
            )
        )

        current_width = max(
            1,
            current_x1 - current_x0
        )

        overlap_ratio = (
            overlap /
            current_width
        )

        same_region = (
            vertical_gap <= 35
            and overlap_ratio >= 0.20
        )

        if same_region:

            current.append(block)

            current_x0 = min(
                current_x0,
                block["x0"]
            )

            current_x1 = max(
                current_x1,
                block["x1"]
            )

        else:

            text = clean_text(
                " ".join(
                    item["text"]
                    for item in current
                )
            )

            if len(text.split()) >= 10:

                articles.append({
                    "text": text,
                    "top": current[0]["y0"],
                    "bottom": current[-1]["y1"]
                })

            current = [block]

            current_x0 = block["x0"]
            current_x1 = block["x1"]

        previous_bottom = block["y1"]

    # Final group
    if current:

        text = clean_text(
            " ".join(
                item["text"]
                for item in current
            )
        )

        if len(text.split()) >= 10:

            articles.append({
                "text": text,
                "top": current[0]["y0"],
                "bottom": current[-1]["y1"]
            })

    return articles


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = image.convert("L")

    width, height = image.size

    target_width = 2200

    if width < target_width:

        ratio = (
            target_width /
            width
        )

        image = image.resize(
            (
                int(width * ratio),
                int(height * ratio)
            ),
            Image.Resampling.LANCZOS
        )

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.5)

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# ============================================================
# OCR EXTRACTION
# ============================================================

def ocr_page_articles(page):

    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            1.4,
            1.4
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

    image = preprocess_image(
        image
    )

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--oem 3 --psm 3"
    )

    grouped = defaultdict(list)

    for i in range(
        len(data["text"])
    ):

        text = data["text"][i].strip()

        if not text:
            continue

        try:

            confidence = float(
                data["conf"][i]
            )

        except:

            confidence = 0

        if confidence < 25:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        grouped[key].append({
            "text": text,
            "x": int(
                data["left"][i]
            ),
            "y": int(
                data["top"][i]
            ),
            "right": (
                int(
                    data["left"][i]
                )
                +
                int(
                    data["width"][i]
                )
            ),
            "bottom": (
                int(
                    data["top"][i]
                )
                +
                int(
                    data["height"][i]
                )
            )
        })

    lines = []

    for words in grouped.values():

        if not words:
            continue

        words.sort(
            key=lambda x:
            x["x"]
        )

        text = clean_text(
            " ".join(
                word["text"]
                for word in words
            )
        )

        if len(text.split()) < 2:
            continue

        lines.append({
            "text": text,
            "x0": min(
                word["x"]
                for word in words
            ),
            "x1": max(
                word["right"]
                for word in words
            ),
            "y0": min(
                word["y"]
                for word in words
            ),
            "y1": max(
                word["bottom"]
                for word in words
            )
        })

    lines.sort(
        key=lambda x: (
            x["y0"],
            x["x0"]
        )
    )

    if not lines:
        return []

    # Simple OCR grouping
    articles = []

    current = []
    previous_y = None

    for line in lines:

        if previous_y is None:

            current = [line]

        else:

            gap = (
                line["y0"]
                -
                previous_y
            )

            if gap <= 45:

                current.append(line)

            else:

                text = clean_text(
                    " ".join(
                        item["text"]
                        for item in current
                    )
                )

                if len(text.split()) >= 10:

                    articles.append({
                        "text": text,
                        "top": current[0]["y0"],
                        "bottom": current[-1]["y1"]
                    })

                current = [line]

        previous_y = line["y1"]

    if current:

        text = clean_text(
            " ".join(
                item["text"]
                for item in current
            )
        )

        if len(text.split()) >= 10:

            articles.append({
                "text": text,
                "top": current[0]["y0"],
                "bottom": current[-1]["y1"]
            })

    return articles


# ============================================================
# RAW PAGE FALLBACK
# ============================================================

def raw_page_fallback(page):

    raw_text = clean_text(
        page.get_text(
            "text",
            sort=True
        )
    )

    if len(raw_text.split()) < 10:
        return []

    normalized = normalize(
        raw_text
    )

    defence_anchors = [

        "military",
        "army",
        "navy",
        "air force",
        "missile",
        "air defence",
        "air defense",
        "defence",
        "defense",
        "troops",
        "fighter",
        "warship",
        "submarine",
        "drone",
        "soldier",
        "ukraine",
        "russia",
        "zelenskyy",
        "zelensky",
        "nato",
        "iran",
        "israel",
        "military operation",
        "armed forces",
        "lavrov",
        "pentagon",
        "kremlin"
    ]

    for keyword in defence_anchors:

        if normalize(keyword) in normalized:

            return [{
                "text": raw_text,
                "top": 0,
                "bottom": page.rect.height
            }]

    return []


# ============================================================
# PROCESS ONE PAGE
# ============================================================

def process_page(
    pdf_bytes,
    page_number
):

    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page = doc.load_page(
        page_number
    )

    # --------------------------------------------------------
    # FAST NATIVE PDF EXTRACTION
    # --------------------------------------------------------

    blocks = extract_pdf_blocks(
        page
    )

    if blocks:

        articles = group_pdf_blocks(
            blocks
        )

        # First try segmented articles
        defence_articles = []

        for article in articles:

            if is_defence_article(
                article["text"]
            ):

                defence_articles.append(
                    article
                )

        if defence_articles:

            doc.close()

            return (
                defence_articles,
                "Native PDF"
            )

        # ----------------------------------------------------
        # RAW PAGE FALLBACK
        # ----------------------------------------------------

        fallback = raw_page_fallback(
            page
        )

        if fallback:

            doc.close()

            return (
                fallback,
                "Native PDF"
            )

    # --------------------------------------------------------
    # OCR FALLBACK
    # --------------------------------------------------------

    try:

        ocr_articles = ocr_page_articles(
            page
        )

        defence_articles = []

        for article in ocr_articles:

            if is_defence_article(
                article["text"]
            ):

                defence_articles.append(
                    article
                )

        if defence_articles:

            doc.close()

            return (
                defence_articles,
                "OCR"
            )

    except Exception:
        pass

    doc.close()

    return [], "None"


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def article_similarity(
    text_a,
    text_b
):

    a = set(
        normalize(text_a).split()
    )

    b = set(
        normalize(text_b).split()
    )

    if not a or not b:
        return 0

    return (
        len(a & b)
        /
        len(a | b)
    )


def remove_duplicates(
    articles
):

    final = []

    for article in articles:

        duplicate = False

        for existing in final:

            similarity = article_similarity(
                article["text"],
                existing["text"]
            )

            if similarity >= 0.70:

                duplicate = True
                break

        if not duplicate:

            final.append(
                article
            )

    return final


# ============================================================
# FINAL DEFENCE FILTER
# ============================================================

def filter_defence(
    articles
):

    result = []

    for article in articles:

        text = article.get(
            "text",
            ""
        )

        if len(text.split()) < 10:
            continue

        if is_defence_article(
            text
        ):

            result.append(
                article
            )

    return result


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


if "file_hash" not in st.session_state:

    st.session_state.file_hash = None


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "AI-assisted extraction of defence-related newspaper articles"
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Newspaper PDF",
    type=["pdf"]
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = file_hash(
        pdf_bytes
    )

    if (
        st.session_state.file_hash
        != current_hash
    ):

        st.session_state.results = []

        st.session_state.file_hash = (
            current_hash
        )

    # Open PDF
    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(
        document
    )

    document.close()

    st.info(
        f"📄 {uploaded_file.name}  |  "
        f"{total_pages} pages"
    )

    # ========================================================
    # PAGE SELECTION
    # ========================================================

    scan_mode = st.radio(
        "Select scan range",
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
                min(
                    3,
                    total_pages
                )
            )
        )

    elif scan_mode == "All pages":

        selected_pages = list(
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

        selected_pages = [
            number - 1
            for number in selected_numbers
        ]

    # ========================================================
    # SCAN BUTTON
    # ========================================================

    if st.button(
        "🚀 SCAN NEWSPAPER",
        type="primary",
        use_container_width=True
    ):

        if not selected_pages:

            st.warning(
                "Please select at least one page."
            )

        else:

            progress = st.progress(
                0
            )

            status = st.empty()

            all_articles = []

            native_pages = 0
            ocr_pages = 0

            # ================================================
            # PAGE-BY-PAGE PROCESSING
            # ================================================

            for index, page_number in enumerate(
                selected_pages
            ):

                status.info(
                    f"🔎 Extracting page "
                    f"{page_number + 1} "
                    f"of {total_pages}..."
                )

                try:

                    articles, method = process_page(
                        pdf_bytes,
                        page_number
                    )

                    if method == "Native PDF":

                        native_pages += 1

                    elif method == "OCR":

                        ocr_pages += 1

                    for article in articles:

                        article["page"] = (
                            page_number + 1
                        )

                    all_articles.extend(
                        articles
                    )

                except Exception as error:

                    st.warning(
                        f"Page {page_number + 1} "
                        f"error: {error}"
                    )

                progress.progress(
                    (
                        index + 1
                    )
                    /
                    len(selected_pages)
                )

            # ================================================
            # REMOVE DUPLICATES
            # ================================================

            all_articles = remove_duplicates(
                all_articles
            )

            # ================================================
            # FINAL FILTER
            # ================================================

            final_articles = filter_defence(
                all_articles
            )

            final_articles.sort(
                key=lambda article: (
                    article.get(
                        "page",
                        0
                    ),
                    article.get(
                        "top",
                        0
                    )
                )
            )

            st.session_state.results = (
                final_articles
            )

            status.success(
                "✅ Newspaper scan completed!"
            )

            # ================================================
            # STATISTICS
            # ================================================

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Pages scanned",
                len(selected_pages)
            )

            col2.metric(
                "Native PDF",
                native_pages
            )

            col3.metric(
                "OCR",
                ocr_pages
            )

            col4.metric(
                "Defence articles",
                len(final_articles)
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

results = st.session_state.results


if results:

    st.divider()

    st.header(
        "📰 Defence News Found"
    )

    st.success(
        f"{len(results)} defence article(s) detected."
    )

    # ========================================================
    # ARTICLE CARDS
    # ========================================================

    for index, article in enumerate(
        results,
        start=1
    ):

        with st.container(
            border=True
        ):

            st.subheader(
                f"News {index}"
            )

            st.caption(
                f"📄 Page {article.get('page', '-')}"
            )

            st.write(
                article["text"]
            )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    output_text = []

    for index, article in enumerate(
        results,
        start=1
    ):

        output_text.append(
            f"NEWS {index}\n"
            f"PAGE {article.get('page', '-')}\n"
            f"{'=' * 80}\n"
            f"{article['text']}\n\n"
        )

    st.download_button(
        "⬇️ Download Extracted Defence News",
        data="\n".join(
            output_text
        ),
        file_name="defence_news.txt",
        mime="text/plain",
        use_container_width=True
    )

elif uploaded_file:

    st.warning(
        "⚠️ No defence articles found in the selected pages."
    )
