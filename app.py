import streamlit as st
import fitz
import re
import hashlib
from collections import defaultdict
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pytesseract import Output


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
# TEXT FUNCTIONS
# ============================================================

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def normalize(text):
    text = text.lower()
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains(text, phrase):
    text = normalize(text)
    phrase = normalize(phrase)

    return re.search(
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])",
        text
    ) is not None


# ============================================================
# DEFENCE FILTER
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

    # Army / Navy / Air Force
    "army",
    "navy",
    "air force",
    "army exercise",
    "naval exercise",
    "air force exercise",
    "troops",
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
    "border security force",
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
]


def is_defence_article(text):

    text = normalize(text)

    if len(text.split()) < 10:
        return False

    matches = 0

    for term in DEFENCE_TERMS:
        if contains(text, term):
            matches += 1

    return matches >= 1


# ============================================================
# PDF NATIVE TEXT EXTRACTION
# ============================================================

def extract_pdf_lines(page):

    page_dict = page.get_text("dict")

    lines = []

    for block in page_dict.get("blocks", []):

        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):

            spans = line.get("spans", [])

            if not spans:
                continue

            parts = []
            sizes = []
            flags = []

            for span in spans:

                txt = span.get(
                    "text",
                    ""
                ).strip()

                if txt:
                    parts.append(txt)

                sizes.append(
                    float(
                        span.get(
                            "size",
                            0
                        )
                    )
                )

                flags.append(
                    int(
                        span.get(
                            "flags",
                            0
                        )
                    )
                )

            text = clean_text(
                " ".join(parts)
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
                "font": max(
                    sizes
                ) if sizes else 0,
                "avg_font": (
                    sum(sizes) / len(sizes)
                    if sizes else 0
                ),
                "bold": any(
                    f & 16
                    for f in flags
                )
            })

    return lines


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(
    lines,
    page_width
):

    if not lines:
        return []

    # Find common left edges.
    left_positions = sorted(
        x["x0"]
        for x in lines
    )

    tolerance = max(
        20,
        page_width * 0.025
    )

    clusters = []

    for x in left_positions:

        found = False

        for cluster in clusters:

            if abs(
                x - cluster["center"]
            ) <= tolerance:

                cluster["values"].append(
                    x
                )

                cluster["center"] = (
                    sum(
                        cluster["values"]
                    )
                    /
                    len(
                        cluster["values"]
                    )
                )

                found = True
                break

        if not found:

            clusters.append({
                "center": x,
                "values": [x]
            })

    # Keep meaningful columns
    useful = []

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

        if count >= 4:

            useful.append(
                cluster
            )

    # If layout is unclear, use page
    # as a single region.
    if len(useful) < 2:

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

    useful.sort(
        key=lambda x:
        x["center"]
    )

    boundaries = []

    for i in range(
        len(useful) - 1
    ):

        boundaries.append(
            (
                useful[i]["center"]
                +
                useful[i + 1]["center"]
            )
            / 2
        )

    columns = []

    for i, cluster in enumerate(
        useful
    ):

        left = (
            0
            if i == 0
            else boundaries[i - 1]
        )

        right = (
            page_width
            if i == len(useful) - 1
            else boundaries[i]
        )

        column_lines = []

        for line in lines:

            center = (
                line["x0"]
                +
                line["x1"]
            ) / 2

            if (
                left
                <= center
                <
                right
            ):

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
                "x0": left,
                "x1": right,
                "lines": column_lines
            })

    return columns


# ============================================================
# HEADLINE DETECTION
# ============================================================

def get_body_font(lines):

    values = []

    for line in lines:

        if len(
            line["text"].split()
        ) >= 5:

            values.append(
                line["avg_font"]
            )

    if not values:
        return 10

    values.sort()

    return values[
        len(values) // 2
    ]


def is_headline(
    line,
    body_font
):

    words = line["text"].split()

    if len(words) < 2:
        return False

    if len(words) > 20:
        return False

    ratio = (
        line["font"]
        /
        max(
            body_font,
            1
        )
    )

    # Large headline
    if ratio >= 1.25:
        return True

    # Bold headline
    if (
        line["bold"]
        and ratio >= 1.08
        and len(words) <= 16
    ):
        return True

    return False


# ============================================================
# ARTICLE SEGMENTATION
# ============================================================

def segment_column(
    column
):

    lines = column["lines"]

    if not lines:
        return []

    body_font = get_body_font(
        lines
    )

    # Detect headline positions
    headlines = []

    for i, line in enumerate(
        lines
    ):

        if is_headline(
            line,
            body_font
        ):

            headlines.append(i)

    articles = []

    # --------------------------------------------------------
    # HEADLINE-BASED EXTRACTION
    # --------------------------------------------------------

    if headlines:

        for n, start in enumerate(
            headlines
        ):

            if n + 1 < len(
                headlines
            ):

                end = headlines[
                    n + 1
                ]

            else:

                end = len(lines)

            group = lines[
                start:end
            ]

            # Don't allow enormous unrelated tails
            group = trim_group(
                group
            )

            text = clean_text(
                " ".join(
                    x["text"]
                    for x in group
                )
            )

            if len(
                text.split()
            ) >= 10:

                articles.append({
                    "text": text,
                    "top": group[0]["y0"],
                    "bottom": group[-1]["y1"]
                })

        return articles

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    current = []

    for line in lines:

        if not current:

            current = [line]
            continue

        previous = current[-1]

        gap = (
            line["y0"]
            -
            previous["y1"]
        )

        # Large physical gap
        if gap > max(
            30,
            previous["height"] * 2.5
        ):

            text = clean_text(
                " ".join(
                    x["text"]
                    for x in current
                )
            )

            if len(
                text.split()
            ) >= 10:

                articles.append({
                    "text": text,
                    "top": current[0]["y0"],
                    "bottom": current[-1]["y1"]
                })

            current = [line]

        else:

            current.append(
                line
            )

    if current:

        text = clean_text(
            " ".join(
                x["text"]
                for x in current
            )
        )

        if len(
            text.split()
        ) >= 10:

            articles.append({
                "text": text,
                "top": current[0]["y0"],
                "bottom": current[-1]["y1"]
            })

    return articles


def trim_group(
    group
):

    output = []

    for line in group:

        text = line["text"].strip()

        if not text:
            continue

        # Remove page-number-only lines
        if re.fullmatch(
            r"\d{1,3}",
            text
        ):
            continue

        output.append(
            line
        )

    return output


# ============================================================
# NATIVE PAGE
# ============================================================

def native_page_articles(
    page
):

    lines = extract_pdf_lines(
        page
    )

    if not lines:
        return []

    total_words = sum(
        len(
            x["text"].split()
        )
        for x in lines
    )

    if total_words < 30:
        return []

    columns = detect_columns(
        lines,
        page.rect.width
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
# OCR PAGE
# ============================================================

def preprocess_image(
    image
):

    image = image.convert(
        "L"
    )

    width, height = image.size

    target_width = 2200

    if width < target_width:

        ratio = (
            target_width
            /
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


def ocr_page_articles(
    page
):

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

        if confidence < 30:
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
            ),
            "height": int(
                data["height"][i]
            )
        })

    lines = []

    for words in grouped.values():

        words.sort(
            key=lambda x:
            x["x"]
        )

        text = clean_text(
            " ".join(
                x["text"]
                for x in words
            )
        )

        if len(
            text.split()
        ) < 2:
            continue

        lines.append({
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
            "bold": False
        })

    if not lines:
        return []

    lines.sort(
        key=lambda x: (
            x["y0"],
            x["x0"]
        )
    )

    columns = detect_columns(
        lines,
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
# PROCESS PDF
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

    # FAST PATH
    native = native_page_articles(
        page
    )

    if native:

        doc.close()

        return native, "Native PDF"

    # OCR ONLY IF NECESSARY
    ocr = ocr_page_articles(
        page
    )

    doc.close()

    return ocr, "OCR"


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def article_similarity(
    a,
    b
):

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


def remove_duplicates(
    articles
):

    final = []

    for article in articles:

        duplicate = False

        for old in final:

            if article_similarity(
                article["text"],
                old["text"]
            ) >= 0.72:

                duplicate = True
                break

        if not duplicate:

            final.append(
                article
            )

    return final


# ============================================================
# FILTER
# ============================================================

def filter_defence(
    articles
):

    result = []

    for article in articles:

        if is_defence_article(
            article["text"]
        ):

            result.append(
                article
            )

    return result


# ============================================================
# FILE HASH
# ============================================================

def file_hash(
    data
):

    return hashlib.md5(
        data
    ).hexdigest()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "hash" not in st.session_state:
    st.session_state.hash = None


# ============================================================
# UI
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "Complete newspaper article extraction and defence filtering"
)

uploaded = st.file_uploader(
    "Upload Newspaper PDF",
    type=["pdf"]
)


if uploaded:

    data = uploaded.getvalue()

    current_hash = file_hash(
        data
    )

    if (
        st.session_state.hash
        != current_hash
    ):

        st.session_state.results = []
        st.session_state.hash = current_hash

    doc = fitz.open(
        stream=data,
        filetype="pdf"
    )

    total_pages = len(doc)

    st.info(
        f"📄 {uploaded.name} | "
        f"{total_pages} pages"
    )

    mode = st.radio(
        "Pages to scan",
        [
            "First 3 pages - TEST",
            "All pages",
            "Select pages"
        ],
        horizontal=True
    )

    if mode == "First 3 pages - TEST":

        selected_pages = list(
            range(
                min(
                    3,
                    total_pages
                )
            )
        )

    elif mode == "All pages":

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
            x - 1
            for x in selected_numbers
        ]

    scan = st.button(
        "🚀 SCAN NEWSPAPER",
        type="primary"
    )

    if scan and selected_pages:

        progress = st.progress(
            0
        )

        status = st.empty()

        all_articles = []

        native_count = 0
        ocr_count = 0

        # ----------------------------------------------------
        # PROCESS SEQUENTIALLY
        # This avoids Streamlit memory overload.
        # ----------------------------------------------------

        for index, page_number in enumerate(
            selected_pages
        ):

            status.info(
                f"🔎 Scanning page "
                f"{page_number + 1} "
                f"of {total_pages}..."
            )

            try:

                articles, method = process_page(
                    data,
                    page_number
                )

                if method == "Native PDF":
                    native_count += 1
                else:
                    ocr_count += 1

                for article in articles:

                    article["page"] = (
                        page_number + 1
                    )

                all_articles.extend(
                    articles
                )

            except Exception as error:

                st.warning(
                    f"Page {page_number + 1}: "
                    f"{error}"
                )

            progress.progress(
                (index + 1)
                /
                len(selected_pages)
            )

        doc.close()

        # Remove duplicates
        all_articles = remove_duplicates(
            all_articles
        )

        # Defence filtering AFTER article extraction
        defence_articles = filter_defence(
            all_articles
        )

        defence_articles.sort(
            key=lambda x: (
                x.get("page", 0),
                x.get("top", 0)
            )
        )

        st.session_state.results = (
            defence_articles
        )

        status.success(
            "✅ Scan completed!"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Pages scanned",
            len(selected_pages)
        )

        c2.metric(
            "Native PDF",
            native_count
        )

        c3.metric(
            "OCR",
            ocr_count
        )


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results


if results:

    st.divider()

    st.header(
        "📰 Defence News"
    )

    st.success(
        f"{len(results)} complete defence article(s) found."
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
                f"Page {article.get('page', '-')}"
            )

            st.write(
                article["text"]
            )

    # Download
    output = []

    for i, article in enumerate(
        results,
        start=1
    ):

        output.append(
            f"NEWS {i}\n"
            f"PAGE {article.get('page', '-')}\n"
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

    st.warning(
        "No defence articles found in the selected pages."
    )
