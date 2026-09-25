
import streamlit as st
import fitz
import re
import hashlib
import statistics
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

try:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"
except Exception:
    pass


# ============================================================
# DEFENCE KEYWORDS
# ============================================================

DEFENCE_TERMS = [
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

    "military",
    "military operation",
    "military operations",
    "military exercise",
    "military exercises",
    "military deployment",
    "military strike",
    "military strikes",
    "military aircraft",
    "military equipment",
    "military base",
    "military training",
    "military commander",
    "military commanders",

    "army",
    "navy",
    "air force",
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

    "defence procurement",
    "defense procurement",
    "defence deal",
    "defense deal",
    "defence contract",
    "defense contract",

    # International defence signals
    "ukraine",
    "russia",
    "zelenskyy",
    "zelensky",
    "nato",
    "pentagon",
    "kremlin",
    "lavrov",
    "iran",
    "israel"
]


# ============================================================
# BASIC TEXT FUNCTIONS
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


def word_count(text):
    return len(text.split())


# ============================================================
# DEFENCE DETECTION
# ============================================================

def defence_score(text):

    text_n = normalize(text)

    score = 0
    matched = []

    for term in DEFENCE_TERMS:

        term_n = normalize(term)

        if term_n in text_n:

            matched.append(term)

            # Stronger terms get more weight
            if term_n in [
                "indian army",
                "indian navy",
                "indian air force",
                "armed forces",
                "military operation",
                "military operations",
                "missile",
                "missiles",
                "air defence",
                "air defense",
                "defence missiles",
                "fighter jet",
                "fighter jets",
                "warship",
                "submarine",
                "artillery",
                "tank",
                "drone",
                "military commander",
                "defence procurement",
                "defence deal"
            ]:
                score += 3
            else:
                score += 1

    return score, matched


def is_defence(text):

    score, matched = defence_score(text)

    if word_count(text) < 12:
        return False

    return score >= 2 or len(matched) >= 1


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def render_page(page):

    pix = page.get_pixmap(
        matrix=fitz.Matrix(
            1.6,
            1.6
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

    return image


def preprocess(image):

    gray = image.convert("L")

    width, height = gray.size

    if width < 1800:

        ratio = 1800 / width

        gray = gray.resize(
            (
                int(width * ratio),
                int(height * ratio)
            ),
            Image.Resampling.LANCZOS
        )

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(1.35)

    gray = gray.filter(
        ImageFilter.SHARPEN
    )

    return gray


# ============================================================
# OCR WORD DATA
# ============================================================

def get_ocr_words(image):

    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--oem 3 --psm 3"
    )

    words = []

    for i in range(
        len(data["text"])
    ):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(
                data["conf"][i]
            )
        except Exception:
            conf = 0

        if conf < 25:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        words.append({
            "text": text,
            "x0": x,
            "x1": x + w,
            "y0": y,
            "y1": y + h,
            "cx": x + w / 2,
            "cy": y + h / 2,
            "height": h,
            "block": data["block_num"][i],
            "par": data["par_num"][i],
            "line": data["line_num"][i]
        })

    return words


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_columns(words, page_width):

    if not words:
        return [
            (0, page_width)
        ]

    centers = sorted(
        [
            w["cx"]
            for w in words
        ]
    )

    if len(centers) < 20:
        return [
            (0, page_width)
        ]

    gaps = []

    for i in range(
        len(centers) - 1
    ):

        gaps.append({
            "gap": centers[i + 1] - centers[i],
            "position": (
                centers[i] +
                centers[i + 1]
            ) / 2
        })

    # Newspaper gutters are usually
    # substantially larger than normal
    # word-to-word gaps.
    median_gap = statistics.median(
        g["gap"]
        for g in gaps
    )

    threshold = max(
        28,
        median_gap * 5,
        page_width * 0.018
    )

    possible = [
        g
        for g in gaps
        if g["gap"] >= threshold
    ]

    # Sort by size
    possible.sort(
        key=lambda x:
        x["gap"],
        reverse=True
    )

    selected = []

    # Maximum 6 newspaper columns
    for candidate in possible:

        pos = candidate["position"]

        # Avoid selecting several nearby
        # gaps belonging to same gutter.
        too_close = False

        for existing in selected:

            if abs(
                pos - existing
            ) < page_width * 0.06:

                too_close = True
                break

        if not too_close:

            selected.append(pos)

        if len(selected) >= 5:
            break

    selected.sort()

    boundaries = []

    for pos in selected:

        # Don't create tiny regions
        if (
            pos > page_width * 0.12
            and
            pos < page_width * 0.88
        ):

            boundaries.append(pos)

    if not boundaries:

        return [
            (0, page_width)
        ]

    columns = []

    start = 0

    for boundary in boundaries:

        if boundary - start >= page_width * 0.12:

            columns.append(
                (
                    start,
                    boundary
                )
            )

            start = boundary

    if page_width - start >= page_width * 0.12:

        columns.append(
            (
                start,
                page_width
            )
        )

    # Sanity check
    if len(columns) < 2:

        return [
            (0, page_width)
        ]

    return columns


# ============================================================
# OCR A SINGLE COLUMN
# ============================================================

def ocr_column(
    image,
    x0,
    x1
):

    crop = image.crop(
        (
            int(x0),
            0,
            int(x1),
            image.height
        )
    )

    data = pytesseract.image_to_data(
        crop,
        output_type=Output.DICT,
        config="--oem 3 --psm 4"
    )

    lines = {}

    for i in range(
        len(data["text"])
    ):

        text = data["text"][i].strip()

        if not text:
            continue

        try:
            conf = float(
                data["conf"][i]
            )
        except Exception:
            conf = 0

        if conf < 25:
            continue

        left = int(
            data["left"][i]
        )

        top = int(
            data["top"][i]
        )

        width = int(
            data["width"][i]
        )

        height = int(
            data["height"][i]
        )

        block = data["block_num"][i]
        par = data["par_num"][i]
        line = data["line_num"][i]

        key = (
            block,
            par,
            line
        )

        if key not in lines:
            lines[key] = []

        lines[key].append({
            "text": text,
            "x": left,
            "y": top,
            "right": left + width,
            "bottom": top + height,
            "height": height
        })

    output = []

    for key, words in lines.items():

        words.sort(
            key=lambda x:
            x["x"]
        )

        text = clean_text(
            " ".join(
                w["text"]
                for w in words
            )
        )

        if not text:
            continue

        output.append({
            "text": text,
            "x0": min(
                w["x"]
                for w in words
            ),
            "x1": max(
                w["right"]
                for w in words
            ),
            "y0": min(
                w["y"]
                for w in words
            ),
            "y1": max(
                w["bottom"]
                for w in words
            ),
            "height": max(
                w["height"]
                for w in words
            )
        })

    output.sort(
        key=lambda x: (
            x["y0"],
            x["x0"]
        )
    )

    return output


# ============================================================
# HEADLINE DETECTION
# ============================================================

def looks_like_headline(line, median_height):

    text = line["text"].strip()

    words = text.split()

    if len(words) < 2:
        return False

    if len(words) > 18:
        return False

    height_ratio = (
        line["height"]
        /
        max(
            median_height,
            1
        )
    )

    # Large OCR font
    if height_ratio >= 1.35:
        return True

    # Short uppercase heading
    letters = re.sub(
        r"[^A-Za-z]",
        "",
        text
    )

    if (
        len(letters) >= 5
        and letters.isupper()
        and len(words) <= 14
    ):
        return True

    # Newspaper-style title
    if (
        len(words) <= 10
        and height_ratio >= 1.15
    ):
        return True

    return False


# ============================================================
# BUILD ARTICLE CANDIDATES INSIDE COLUMN
# ============================================================

def build_column_articles(lines):

    if not lines:
        return []

    heights = [
        x["height"]
        for x in lines
        if x["height"] > 0
    ]

    median_height = (
        statistics.median(heights)
        if heights
        else 15
    )

    # --------------------------------------------------------
    # Mark probable headlines
    # --------------------------------------------------------

    headline_indexes = []

    for i, line in enumerate(lines):

        if looks_like_headline(
            line,
            median_height
        ):

            headline_indexes.append(i)

    # --------------------------------------------------------
    # Create sections using headlines
    # --------------------------------------------------------

    sections = []

    if headline_indexes:

        for n, start in enumerate(
            headline_indexes
        ):

            if n + 1 < len(
                headline_indexes
            ):

                end = headline_indexes[
                    n + 1
                ]

            else:

                end = len(lines)

            section = lines[
                start:end
            ]

            if section:

                sections.append(
                    section
                )

    else:

        # Fallback based on vertical gaps
        current = []

        previous_bottom = None

        for line in lines:

            if previous_bottom is None:

                current = [line]

            else:

                gap = (
                    line["y0"]
                    -
                    previous_bottom
                )

                if gap > median_height * 3:

                    if current:
                        sections.append(
                            current
                        )

                    current = [line]

                else:

                    current.append(
                        line
                    )

            previous_bottom = line["y1"]

        if current:
            sections.append(
                current
            )

    # --------------------------------------------------------
    # Convert sections to text
    # --------------------------------------------------------

    articles = []

    for section in sections:

        text = clean_text(
            " ".join(
                line["text"]
                for line in section
            )
        )

        if word_count(text) < 12:
            continue

        articles.append({
            "text": text,
            "top": section[0]["y0"],
            "bottom": section[-1]["y1"]
        })

    return articles


# ============================================================
# EXPAND DEFENCE ARTICLE
# ============================================================

def expand_defence_article(
    article,
    all_lines,
    index
):

    start = index
    end = index

    # Include preceding headline/line
    # if it is very close.
    if index > 0:

        previous = all_lines[
            index - 1
        ]

        current = all_lines[
            index
        ]

        gap = (
            current["y0"]
            -
            previous["y1"]
        )

        if gap < current["height"] * 2.5:

            start = index - 1

    # Expand downward through nearby
    # paragraphs, but stop at obvious
    # new headlines.
    median_height = statistics.median(
        [
            x["height"]
            for x in all_lines
            if x["height"] > 0
        ]
    ) if all_lines else 15

    for j in range(
        end + 1,
        len(all_lines)
    ):

        previous = all_lines[
            j - 1
        ]

        current = all_lines[
            j
        ]

        gap = (
            current["y0"]
            -
            previous["y1"]
        )

        if (
            looks_like_headline(
                current,
                median_height
            )
            and
            j > index
        ):

            break

        if gap > median_height * 3:

            break

        end = j

    selected = all_lines[
        start:end + 1
    ]

    text = clean_text(
        " ".join(
            x["text"]
            for x in selected
        )
    )

    return {
        "text": text,
        "top": selected[0]["y0"],
        "bottom": selected[-1]["y1"]
    }


# ============================================================
# DEFENCE ARTICLES FROM COLUMN
# ============================================================

def extract_defence_from_column(
    lines
):

    if not lines:
        return []

    results = []

    # Find lines containing defence signals
    for i, line in enumerate(lines):

        score, matched = defence_score(
            line["text"]
        )

        if score <= 0:
            continue

        candidate = expand_defence_article(
            None,
            lines,
            i
        )

        text = candidate["text"]

        if word_count(text) < 12:
            continue

        final_score, final_matches = (
            defence_score(text)
        )

        if final_score >= 2:

            candidate["matches"] = (
                final_matches
            )

            results.append(
                candidate
            )

    # --------------------------------------------------------
    # Merge overlapping results
    # --------------------------------------------------------

    merged = []

    for article in results:

        duplicate = False

        for existing in merged:

            if (
                article["top"]
                <= existing["bottom"]
                and
                article["bottom"]
                >= existing["top"]
            ):

                # Keep the larger article
                if word_count(
                    article["text"]
                ) > word_count(
                    existing["text"]
                ):

                    existing["text"] = (
                        article["text"]
                    )

                    existing["bottom"] = (
                        article["bottom"]
                    )

                duplicate = True
                break

        if not duplicate:

            merged.append(
                article
            )

    return merged


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

    image = render_page(
        page
    )

    image = preprocess(
        image
    )

    # --------------------------------------------------------
    # FIRST OCR PASS
    # Used ONLY to determine columns
    # --------------------------------------------------------

    words = get_ocr_words(
        image
    )

    if not words:

        doc.close()

        return [], "OCR failed"

    columns = detect_columns(
        words,
        image.width
    )

    all_articles = []

    # --------------------------------------------------------
    # OCR EACH COLUMN SEPARATELY
    # --------------------------------------------------------

    for column_index, (
        x0,
        x1
    ) in enumerate(columns):

        # Small overlap prevents text touching
        # an exact gutter from disappearing.
        overlap = 5

        crop_x0 = max(
            0,
            x0 - overlap
        )

        crop_x1 = min(
            image.width,
            x1 + overlap
        )

        lines = ocr_column(
            image,
            crop_x0,
            crop_x1
        )

        if not lines:
            continue

        articles = (
            extract_defence_from_column(
                lines
            )
        )

        for article in articles:

            article["column"] = (
                column_index + 1
            )

            all_articles.append(
                article
            )

    doc.close()

    return all_articles, "Column OCR"


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def similarity(
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

            if similarity(
                article["text"],
                existing["text"]
            ) >= 0.65:

                duplicate = True

                # Keep longer version
                if word_count(
                    article["text"]
                ) > word_count(
                    existing["text"]
                ):

                    existing["text"] = (
                        article["text"]
                    )

                    existing["top"] = (
                        article["top"]
                    )

                    existing["bottom"] = (
                        article["bottom"]
                    )

                break

        if not duplicate:

            final.append(
                article
            )

    return final


# ============================================================
# FILE HASH
# ============================================================

def get_hash(data):

    return hashlib.md5(
        data
    ).hexdigest()


# ============================================================
# SESSION
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# UI
# ============================================================

st.title(
    "🛡️ Defence News Scanner OCR"
)

st.caption(
    "Layout-aware newspaper OCR for complete defence news articles"
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

        st.session_state.results = []

        st.session_state.file_hash = (
            current_hash
        )

    # Get page count
    doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(doc)

    doc.close()

    st.info(
        f"📄 {uploaded.name} | "
        f"{total_pages} pages"
    )

    # --------------------------------------------------------
    # SCAN RANGE
    # --------------------------------------------------------

    mode = st.radio(
        "Select scan range",
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
            "Select pages",
            list(
                range(
                    1,
                    total_pages + 1
                )
            )
        )

        selected_pages = [
            n - 1
            for n in selected_numbers
        ]

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    if st.button(
        "🚀 SCAN NEWSPAPER",
        type="primary",
        use_container_width=True
    ):

        if not selected_pages:

            st.warning(
                "Select at least one page."
            )

        else:

            progress = st.progress(
                0
            )

            status = st.empty()

            all_results = []

            for count, page_number in enumerate(
                selected_pages
            ):

                status.info(
                    f"🔎 Reading page "
                    f"{page_number + 1} "
                    f"of {total_pages}..."
                )

                try:

                    articles, method = (
                        process_page(
                            pdf_bytes,
                            page_number
                        )
                    )

                    for article in articles:

                        article["page"] = (
                            page_number + 1
                        )

                    all_results.extend(
                        articles
                    )

                except Exception as e:

                    st.warning(
                        f"Page "
                        f"{page_number + 1}: "
                        f"{str(e)}"
                    )

                progress.progress(
                    (
                        count + 1
                    )
                    /
                    len(selected_pages)
                )

            # Remove duplicates
            all_results = (
                remove_duplicates(
                    all_results
                )
            )

            # Sort by page/column/position
            all_results.sort(
                key=lambda x: (
                    x.get("page", 0),
                    x.get("column", 0),
                    x.get("top", 0)
                )
            )

            st.session_state.results = (
                all_results
            )

            status.success(
                "✅ Scan completed!"
            )

            st.success(
                f"Found {len(all_results)} "
                f"defence-related article(s)."
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

            st.caption(
                f"📄 Page {article.get('page', '-')}"
                f"  •  Column {article.get('column', '-')}"
            )

            # Matched defence signals
            matches = article.get(
                "matches",
                []
            )

            if matches:

                shown = ", ".join(
                    matches[:6]
                )

                st.caption(
                    f"🔎 Defence signals: {shown}"
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
            f"COLUMN: {article.get('column', '-')}\n"
            f"{'=' * 80}\n"
            f"{article['text']}\n\n"
        )

    st.download_button(
        "⬇️ Download Defence News",
        data="".join(output),
        file_name="defence_news.txt",
        mime="text/plain",
        use_container_width=True
    )

elif uploaded:

    st.warning(
        "No defence articles found."
    )
