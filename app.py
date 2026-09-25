import streamlit as st
import fitz
import re
import hashlib
from collections import defaultdict
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Defence News Scanner OCR",
    page_icon="🛡️",
    layout="wide"
)

pytesseract.pytesseract.tesseract_cmd = "tesseract"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def norm(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_exists(text, phrase):
    text = norm(text)
    phrase = norm(phrase)

    return re.search(
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])",
        text
    ) is not None


# ============================================================
# STRICT DEFENCE CLASSIFICATION
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
    "defence minister",
    "defense minister",
    "chief of defence staff",
    "chief of defense staff",
    "chief of army staff",
    "chief of naval staff",
    "chief of air staff",
    "army chief",
    "navy chief",
    "air chief",
    "army exercise",
    "naval exercise",
    "air force exercise",
    "military exercise",
    "military exercises",
    "military operation",
    "military operations",
    "military deployment",
    "military strike",
    "military strikes",
    "defence procurement",
    "defense procurement",
    "defence deal",
    "defense deal",
    "defence contract",
    "defense contract",
]

HARDWARE = [
    "missile",
    "missiles",
    "ballistic missile",
    "cruise missile",
    "air defence system",
    "air defense system",
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
    "drone strike",
    "drone strikes",
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
    "military aircraft",
    "military equipment",
    "radar",
]

MILITARY_CONTEXT = [
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
    "combat operations",
    "warfare",
    "military training",
    "military deployment",
    "military operation",
    "military operations",
    "air strike",
    "airstrike",
    "airstrikes",
    "missile strike",
    "missile strikes",
]

SECURITY = [
    "border security force",
    "border security",
    "border guards",
    "border patrol",
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
]

CONFLICT = [
    "military conflict",
    "armed conflict",
    "invasion",
    "warfare",
    "airstrike",
    "airstrikes",
    "missile strike",
    "missile strikes",
    "drone strike",
    "drone strikes",
]


def is_defence_article(article):

    text = norm(article)

    if len(text.split()) < 25:
        return False

    direct = sum(
        phrase_exists(text, x)
        for x in DIRECT_DEFENCE
    )

    hardware = sum(
        phrase_exists(text, x)
        for x in HARDWARE
    )

    military = sum(
        phrase_exists(text, x)
        for x in MILITARY_CONTEXT
    )

    security = sum(
        phrase_exists(text, x)
        for x in SECURITY
    )

    conflict = sum(
        phrase_exists(text, x)
        for x in CONFLICT
    )

    # Direct defence article
    if direct >= 1:
        return True

    # Weapon/equipment + military context
    if hardware >= 1 and military >= 1:
        return True

    # Weapon/equipment + security context
    if hardware >= 1 and security >= 1:
        return True

    # Multiple military hardware terms
    if hardware >= 2:
        return True

    # Strong security story
    if security >= 2:
        return True

    # Actual military conflict
    if conflict >= 1 and (
        military >= 1
        or hardware >= 1
        or security >= 1
    ):
        return True

    return False


# ============================================================
# PDF LINE EXTRACTION
# ============================================================

def extract_pdf_lines(page):

    data = page.get_text("dict")

    lines = []

    for block in data.get("blocks", []):

        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):

            spans = line.get("spans", [])

            if not spans:
                continue

            text_parts = []

            sizes = []
            flags = []

            for span in spans:

                txt = span.get("text", "").strip()

                if txt:
                    text_parts.append(txt)

                sizes.append(
                    float(span.get("size", 0))
                )

                flags.append(
                    int(span.get("flags", 0))
                )

            text = clean_text(
                " ".join(text_parts)
            )

            if not text:
                continue

            bbox = line.get(
                "bbox",
                [0, 0, 0, 0]
            )

            lines.append({
                "text": text,
                "x0": bbox[0],
                "y0": bbox[1],
                "x1": bbox[2],
                "y1": bbox[3],
                "height": max(
                    1,
                    bbox[3] - bbox[1]
                ),
                "font": max(sizes) if sizes else 0,
                "avg_font": (
                    sum(sizes) / len(sizes)
                    if sizes else 0
                ),
                "bold": any(
                    f & 16
                    for f in flags
                ),
            })

    return lines


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_page_columns(lines, page_width):

    if not lines:
        return []

    # We create horizontal "bands" using the actual
    # left positions of newspaper text.
    #
    # This prevents:
    #
    # LEFT COLUMN ARTICLE
    # +
    # MIDDLE COLUMN ARTICLE
    # +
    # RIGHT COLUMN ARTICLE
    #
    # from becoming one text stream.

    starts = sorted(
        set(
            round(x["x0"], 1)
            for x in lines
            if x["x1"] - x["x0"] > 10
        )
    )

    clusters = []

    tolerance = max(
        18,
        page_width * 0.025
    )

    for start in starts:

        placed = False

        for cluster in clusters:

            if abs(
                start - cluster["center"]
            ) <= tolerance:

                cluster["starts"].append(
                    start
                )

                cluster["center"] = sum(
                    cluster["starts"]
                ) / len(
                    cluster["starts"]
                )

                placed = True
                break

        if not placed:

            clusters.append({
                "center": start,
                "starts": [start]
            })

    # Ignore tiny accidental x clusters
    valid_clusters = []

    for cluster in clusters:

        count = sum(
            1
            for line in lines
            if abs(
                line["x0"]
                -
                cluster["center"]
            ) <= tolerance
        )

        if count >= 3:
            valid_clusters.append(
                cluster
            )

    # If column detection failed,
    # use one full page region.
    if len(valid_clusters) <= 1:

        return [{
            "x0": 0,
            "x1": page_width,
            "lines": sorted(
                lines,
                key=lambda x: (
                    x["y0"],
                    x["x0"]
                )
            )
        }]

    # Create boundaries halfway between columns
    valid_clusters.sort(
        key=lambda x: x["center"]
    )

    boundaries = []

    for i in range(
        len(valid_clusters) - 1
    ):

        left = valid_clusters[i]["center"]
        right = valid_clusters[i + 1]["center"]

        boundaries.append(
            (left + right) / 2
        )

    columns = []

    for i, cluster in enumerate(
        valid_clusters
    ):

        if i == 0:
            x0 = 0
        else:
            x0 = boundaries[i - 1]

        if i == len(valid_clusters) - 1:
            x1 = page_width
        else:
            x1 = boundaries[i]

        column_lines = []

        for line in lines:

            center = (
                line["x0"]
                +
                line["x1"]
            ) / 2

            if x0 <= center < x1:
                column_lines.append(
                    line
                )

        if column_lines:

            column_lines.sort(
                key=lambda x: (
                    x["y0"],
                    x["x0"]
                )
            )

            columns.append({
                "x0": x0,
                "x1": x1,
                "lines": column_lines
            })

    return columns


# ============================================================
# HEADLINE DETECTION
# ============================================================

def determine_body_font(lines):

    fonts = []

    for line in lines:

        words = len(
            line["text"].split()
        )

        # Normal article body is usually
        # reasonably long.
        if words >= 5:
            fonts.append(
                line["avg_font"]
            )

    if not fonts:
        return 10

    fonts.sort()

    return fonts[
        len(fonts) // 2
    ]


def is_probable_headline(
    line,
    body_font
):

    text = line["text"]
    words = text.split()

    if len(words) < 2:
        return False

    if len(words) > 25:
        return False

    font_ratio = (
        line["font"]
        /
        max(body_font, 1)
    )

    # Large headline
    if font_ratio >= 1.28:
        return True

    # Bold + somewhat larger than body
    if (
        line["bold"]
        and font_ratio >= 1.10
        and len(words) <= 18
    ):
        return True

    # Short bold headline
    if (
        line["bold"]
        and len(words) <= 12
    ):
        return True

    return False


# ============================================================
# ARTICLE SEGMENTATION
# ============================================================

def segment_column(column):

    lines = column["lines"]

    if not lines:
        return []

    body_font = determine_body_font(
        lines
    )

    headline_indices = []

    for i, line in enumerate(lines):

        if is_probable_headline(
            line,
            body_font
        ):

            headline_indices.append(i)

    # If there are too few headlines,
    # use spatial blocks instead of forcing
    # everything into one article.
    if len(headline_indices) == 0:

        return fallback_spatial_segmentation(
            lines
        )

    articles = []

    for position, start_index in enumerate(
        headline_indices
    ):

        if position + 1 < len(
            headline_indices
        ):

            next_index = headline_indices[
                position + 1
            ]

        else:

            next_index = len(lines)

        article_lines = lines[
            start_index:next_index
        ]

        if not article_lines:
            continue

        # Remove accidental huge blank-space tail
        article_lines = trim_article_lines(
            article_lines
        )

        if not article_lines:
            continue

        text = clean_text(
            " ".join(
                x["text"]
                for x in article_lines
            )
        )

        if len(text.split()) >= 25:

            articles.append({
                "text": text,
                "top": article_lines[0]["y0"],
                "bottom": article_lines[-1]["y1"],
                "x0": column["x0"],
                "x1": column["x1"],
            })

    return articles


# ============================================================
# FALLBACK SPATIAL SEGMENTATION
# ============================================================

def fallback_spatial_segmentation(lines):

    articles = []

    current = []

    for i, line in enumerate(lines):

        if not current:

            current = [line]
            continue

        previous = current[-1]

        gap = (
            line["y0"]
            -
            previous["y1"]
        )

        # Large blank area = new article
        if gap > max(
            24,
            previous["height"] * 2.2
        ):

            text = clean_text(
                " ".join(
                    x["text"]
                    for x in current
                )
            )

            if len(text.split()) >= 25:

                articles.append({
                    "text": text,
                    "top": current[0]["y0"],
                    "bottom": current[-1]["y1"],
                    "x0": current[0]["x0"],
                    "x1": current[-1]["x1"],
                })

            current = [line]

        else:

            current.append(line)

    if current:

        text = clean_text(
            " ".join(
                x["text"]
                for x in current
            )
        )

        if len(text.split()) >= 25:

            articles.append({
                "text": text,
                "top": current[0]["y0"],
                "bottom": current[-1]["y1"],
                "x0": current[0]["x0"],
                "x1": current[-1]["x1"],
            })

    return articles


# ============================================================
# TRIM BAD ARTICLE TAILS
# ============================================================

def trim_article_lines(lines):

    cleaned = []

    for line in lines:

        text = line["text"].strip()

        if not text:
            continue

        # Page-number-only lines
        if re.fullmatch(
            r"\d{1,3}",
            text
        ):
            continue

        # Website-only lines
        if re.fullmatch(
            r"(www\.)?[a-z0-9.-]+\.(com|in|org|net)",
            text.lower()
        ):
            continue

        cleaned.append(line)

    return cleaned


# ============================================================
# EXTRACT ONE PDF PAGE
# ============================================================

def extract_page_articles(page):

    lines = extract_pdf_lines(
        page
    )

    if not lines:
        return []

    # If native PDF has very little usable text,
    # caller will use OCR.
    total_words = sum(
        len(x["text"].split())
        for x in lines
    )

    if total_words < 50:
        return []

    columns = detect_page_columns(
        lines,
        page.rect.width
    )

    articles = []

    for column in columns:

        column_articles = segment_column(
            column
        )

        articles.extend(
            column_articles
        )

    return articles


# ============================================================
# OCR FALLBACK
# ============================================================

def preprocess_image(image):

    image = image.convert("L")

    width, height = image.size

    target_width = 2200

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

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.5)

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


def ocr_page_articles(page):

    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            1.5,
            1.5
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

    lines = defaultdict(list)

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

        if confidence < 35:
            continue

        block = data["block_num"][i]
        paragraph = data["par_num"][i]
        line_no = data["line_num"][i]

        key = (
            block,
            paragraph,
            line_no
        )

        lines[key].append({
            "text": text,
            "x": int(
                data["left"][i]
            ),
            "y": int(
                data["top"][i]
            ),
            "right": (
                int(data["left"][i])
                +
                int(data["width"][i])
            ),
            "bottom": (
                int(data["top"][i])
                +
                int(data["height"][i])
            ),
            "height": int(
                data["height"][i]
            ),
        })

    ocr_lines = []

    for words in lines.values():

        words.sort(
            key=lambda x: x["x"]
        )

        text = clean_text(
            " ".join(
                x["text"]
                for x in words
            )
        )

        if len(text.split()) < 2:
            continue

        ocr_lines.append({
            "text": text,
            "x0": min(
                x["x"]
                for x in words
            ),
            "x1": max(
                x["right"]
                for x in words
            ),
            "y0": min(
                x["y"]
                for x in words
            ),
            "y1": max(
                x["bottom"]
                for x in words
            ),
            "height": max(
                x["height"]
                for x in words
            ),
            "font": max(
                x["height"]
                for x in words
            ),
            "avg_font": sum(
                x["height"]
                for x in words
            ) / len(words),
            "bold": False,
        })

    if not ocr_lines:
        return []

    ocr_lines.sort(
        key=lambda x: (
            x["y0"],
            x["x0"]
        )
    )

    columns = detect_page_columns(
        ocr_lines,
        image.width
    )

    articles = []

    for column in columns:

        articles.extend(
            segment_column(
                column
            )
        )

    return articles


# ============================================================
# PAGE PROCESSOR
# ============================================================

def process_page(
    pdf_bytes,
    page_number
):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    page = document.load_page(
        page_number
    )

    # FIRST: native text
    articles = extract_page_articles(
        page
    )

    if articles:

        document.close()

        return articles, "native"

    # ONLY FALL BACK TO OCR IF NATIVE
    # TEXT IS NOT USABLE.
    articles = ocr_page_articles(
        page
    )

    document.close()

    return articles, "ocr"


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def similarity(a, b):

    a = set(
        norm(a).split()
    )

    b = set(
        norm(b).split()
    )

    if not a or not b:
        return 0

    return len(
        a & b
    ) / len(
        a | b
    )


def remove_duplicates(articles):

    final = []

    for article in articles:

        duplicate = False

        for existing in final:

            if similarity(
                article["text"],
                existing["text"]
            ) >= 0.70:

                duplicate = True
                break

        if not duplicate:

            final.append(
                article
            )

    return final


# ============================================================
# DEFENCE FILTER
# ============================================================

def filter_defence(
    articles
):

    result = []

    for article in articles:

        text = article["text"]

        if is_defence_article(
            text
        ):

            result.append(
                article
            )

    return result


# ============================================================
# HASH
# ============================================================

def get_hash(data):

    return hashlib.md5(
        data
    ).hexdigest()


# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# UI
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "Complete article extraction → defence relevance filtering"
)

uploaded = st.file_uploader(
    "Upload Newspaper PDF",
    type=["pdf"]
)


if uploaded:

    pdf_bytes = uploaded.getvalue()

    current_hash = get_hash(
        pdf_bytes
    )

    if (
        st.session_state.file_hash
        != current_hash
    ):

        st.session_state.scan_results = []
        st.session_state.file_hash = current_hash

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf)

    st.info(
        f"📄 {uploaded.name} | "
        f"{total_pages} pages"
    )

    scan_mode = st.radio(
        "Scan mode",
        [
            "First 3 pages — TEST",
            "All pages",
            "Select pages"
        ],
        horizontal=True
    )

    if scan_mode == "First 3 pages — TEST":

        pages = list(
            range(
                min(
                    3,
                    total_pages
                )
            )
        )

    elif scan_mode == "All pages":

        pages = list(
            range(
                total_pages
            )
        )

    else:

        selected = st.multiselect(
            "Select pages",
            list(
                range(
                    1,
                    total_pages + 1
                )
            )
        )

        pages = [
            x - 1
            for x in selected
        ]

    scan_button = st.button(
        "🚀 SCAN NEWSPAPER",
        type="primary"
    )

    if scan_button and pages:

        all_articles = []

        native_count = 0
        ocr_count = 0

        progress = st.progress(
            0
        )

        status = st.empty()

        # ----------------------------------------------------
        # FAST NATIVE PROCESSING
        #
        # Only OCR pages that actually need OCR.
        # ----------------------------------------------------

        with ThreadPoolExecutor(
            max_workers=min(
                4,
                len(pages)
            )
        ) as executor:

            futures = {
                executor.submit(
                    process_page,
                    pdf_bytes,
                    page_no
                ): page_no
                for page_no in pages
            }

            completed = 0

            for future in as_completed(
                futures
            ):

                page_no = futures[
                    future
                ]

                try:

                    articles, method = (
                        future.result()
                    )

                    if method == "native":
                        native_count += 1
                    else:
                        ocr_count += 1

                    for article in articles:

                        article["page"] = (
                            page_no + 1
                        )

                    all_articles.extend(
                        articles
                    )

                except Exception as e:

                    st.warning(
                        f"Page {page_no + 1}: {e}"
                    )

                completed += 1

                progress.progress(
                    completed / len(pages)
                )

                status.info(
                    f"Processing "
                    f"{completed}/{len(pages)} pages..."
                )

        pdf.close()

        # ----------------------------------------------------
        # DEDUP
        # ----------------------------------------------------

        all_articles = remove_duplicates(
            all_articles
        )

        # ----------------------------------------------------
        # DEFENCE FILTER
        # ----------------------------------------------------

        defence_articles = filter_defence(
            all_articles
        )

        # Sort by page
        defence_articles.sort(
            key=lambda x: (
                x.get("page", 0),
                x.get("top", 0)
            )
        )

        st.session_state.scan_results = (
            defence_articles
        )

        status.success(
            f"✅ Completed — "
            f"{len(defence_articles)} "
            f"defence article(s) found."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Pages",
            len(pages)
        )

        c2.metric(
            "Native",
            native_count
        )

        c3.metric(
            "OCR",
            ocr_count
        )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.scan_results


if results:

    st.divider()

    st.header(
        "📰 Defence News"
    )

    st.write(
        f"**{len(results)} complete article(s)**"
    )

    for number, article in enumerate(
        results,
        start=1
    ):

        with st.container(
            border=True
        ):

            st.markdown(
                f"### News {number}"
            )

            st.caption(
                f"Newspaper page {article.get('page', '-')}"
            )

            st.write(
                article["text"]
            )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    output = []

    for number, article in enumerate(
        results,
        start=1
    ):

        output.append(
            f"NEWS {number}\n"
            f"PAGE: {article.get('page', '-')}\n"
            f"{'=' * 80}\n"
            f"{article['text']}\n\n"
        )

    st.download_button(
        "⬇️ Download Defence News",
        data="\n".join(output),
        file_name="defence_news.txt",
        mime="text/plain"
    )

elif uploaded:

    st.info(
        "No defence-related complete articles "
        "were detected in the selected pages."
    )
