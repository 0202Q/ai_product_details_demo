# app.py
from __future__ import annotations

import json
import re
import html
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
from PIL import Image


# =============================
# Demo constants (simulated API)
# =============================

DEMO_VENDOR_URL = "https://www.aliexpress.us/item/3256810240290544.html"

# Simulated "AliExpress API" payload (reliable demo — no scraping)
DEMO_API_PAYLOAD = {
    "vendor_url": DEMO_VENDOR_URL,
    "raw_title": "Solid Color Slim 2025 Autumn Elegant Trend Tight Women's Long Sleeve Sexy Bodysuit Outfits Female Tops BKK1196-1",
    "vendor_color": "BKK1196-1",
    "variants_text": (
        "Color: BKK1196-1, Size: S\n"
        "Color: BKK1196-1, Size: M\n"
        "Color: BKK1196-1, Size: L\n"
        "Color: BKK1196-1, Size: XL"
    ),
    # Simulated "size chart via API"
    "size_chart_json": {
        "unit": "cm",
        "columns": ["Size", "Bust", "Shoulder", "Sleeve", "Waist", "Hip", "Length"],
        "rows": [
            ["S", 74, 33, 58, 60, 74, 72],
            ["M", 78, 34, 59, 64, 78, 73],
            ["L", 82, 35, 60, 68, 82, 74],
            ["XL", 86, 36, 61, 72, 86, 75],
        ],
    },
}


# =============================
# Rules you asked for
# =============================

TITLE_MAX_CHARS = 60
FORBIDDEN_WORDS = {"women", "woman", "female", "girls", "lady", "ladies"}
# no punctuation, no hyphens: keep letters/numbers/spaces only
TITLE_ALLOWED_RE = re.compile(r"[^A-Za-z0-9 ]+")

# vendor style codes like BKK1196-1
STYLE_CODE_RE = re.compile(r"\b[A-Z]{2,}[0-9][A-Z0-9]*([_-]?[A-Z0-9]+)*\b")

COLOR_CANONICAL = {
    "black": "Black",
    "white": "White",
    "ivory": "Ivory",
    "cream": "Cream",
    "beige": "Beige",
    "brown": "Brown",
    "tan": "Tan",
    "camel": "Camel",
    "navy": "Navy",
    "blue": "Blue",
    "light blue": "Light Blue",
    "sky blue": "Sky Blue",
    "green": "Green",
    "light green": "Light Green",
    "sage": "Sage",
    "olive": "Olive",
    "red": "Red",
    "burgundy": "Burgundy",
    "wine": "Burgundy",
    "pink": "Pink",
    "purple": "Purple",
    "lavender": "Lavender",
    "yellow": "Yellow",
    "orange": "Orange",
    "gray": "Gray",
    "grey": "Gray",
    "silver": "Silver",
    "gold": "Gold",
    "khaki": "Khaki",
}


# =============================
# Helpers (cleaning + parsing)
# =============================

def standardize_color(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return COLOR_CANONICAL.get(s, raw.strip())


def looks_like_vendor_code(s: str) -> bool:
    if not s:
        return False
    return bool(STYLE_CODE_RE.search(s.strip()))


def strip_vendor_codes(text: str) -> str:
    if not text:
        return ""
    out = STYLE_CODE_RE.sub(" ", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def remove_forbidden_words(text: str) -> str:
    if not text:
        return ""
    words = text.split()
    kept = [w for w in words if w.lower() not in FORBIDDEN_WORDS]
    return " ".join(kept)


def enforce_title_rules(title: str) -> str:
    title = title.strip()

    # remove style codes + forbidden words
    title = strip_vendor_codes(title)
    title = remove_forbidden_words(title)

    # remove punctuation + hyphens (and anything else not allowed)
    title = TITLE_ALLOWED_RE.sub(" ", title)
    title = re.sub(r"\s+", " ", title).strip()

    # hard limit length
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS].rstrip()

        # avoid ending mid-word if we can
        if " " in title:
            title = title.rsplit(" ", 1)[0].rstrip()

    return title


def parse_variants(variants_text: str) -> List[Dict[str, str]]:
    """
    Parses lines like: "Color: X, Size: M"
    """
    out: List[Dict[str, str]] = []
    if not variants_text:
        return out

    for line in variants_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # flexible parsing
        m_color = re.search(r"color\s*:\s*([^,]+)", line, re.I)
        m_size = re.search(r"size\s*:\s*([A-Za-z0-9]+)", line, re.I)
        color = m_color.group(1).strip() if m_color else ""
        size = m_size.group(1).strip().upper() if m_size else ""

        if size:
            out.append({"color": color, "size": size})

    return out


def clean_size_chart(size_chart_json: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Returns (markdown_table, warnings)
    """
    warnings: List[str] = []
    if not size_chart_json:
        return ("", ["Missing size chart JSON."])

    unit = str(size_chart_json.get("unit", "")).strip() or "cm"
    cols = size_chart_json.get("columns", [])
    rows = size_chart_json.get("rows", [])

    if not isinstance(cols, list) or not cols:
        return ("", ["Size chart columns missing/invalid."])
    if not isinstance(rows, list) or not rows:
        return ("", ["Size chart rows missing/invalid."])

    # Ensure Size is first column
    if cols[0].lower() != "size":
        warnings.append("First column is not 'Size' — leaving as-is.")

    # Build markdown table
    header = "| " + " | ".join([str(c) for c in cols]) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    body_lines = []
    for r in rows:
        if not isinstance(r, list):
            continue
        # pad/truncate to match columns length
        rr = (r + [""] * len(cols))[: len(cols)]
        body_lines.append("| " + " | ".join([str(x) for x in rr]) + " |")

    table = "\n".join([header, sep] + body_lines)
    table = f"Unit: {unit}\n\n{table}"
    return (table, warnings)


# =============================
# "Vision" (demo-friendly)
# =============================

def vision_extract_color(img: Image.Image) -> Tuple[str, str]:
    """
    Simple heuristic (demo). Returns (color_name, confidence).
    If you swap to a real Vision API later, keep this interface.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    # center crop to reduce background
    crop = img.crop((int(w * 0.25), int(h * 0.20), int(w * 0.75), int(h * 0.85)))
    # downsample
    crop = crop.resize((64, 64))

    px = list(crop.getdata())
    avg_r = sum(p[0] for p in px) / len(px)
    avg_g = sum(p[1] for p in px) / len(px)
    avg_b = sum(p[2] for p in px) / len(px)

    # brightness
    bright = (avg_r + avg_g + avg_b) / 3.0

    # crude classification
    if bright < 60:
        return ("Black", "High")
    if bright > 200:
        return ("White", "Medium")

    # dominant channel
    if avg_g > avg_r + 15 and avg_g > avg_b + 15:
        return ("Green", "Medium")
    if avg_r > avg_g + 15 and avg_r > avg_b + 15:
        return ("Red", "Medium")
    if avg_b > avg_r + 15 and avg_b > avg_g + 15:
        return ("Blue", "Medium")

    return ("Black", "Low")  # safe default for your demo case


# =============================
# Listing generation (simulated LLM)
# =============================

def infer_product_type(raw_title: str) -> str:
    t = (raw_title or "").lower()
    if "bodysuit" in t:
        return "Bodysuit Top"
    if "romper" in t:
        return "Romper"
    if "dress" in t:
        return "Dress"
    return "Top"


def build_clean_title(product_type: str, color: str, raw_title: str) -> str:
    """
    Title <= 60 chars, no punctuation/hyphens, no 'women', no vendor codes.
    """
    color = standardize_color(color) or "Black"

    # Pull a few useful attributes from raw title (lightweight + safe)
    rt = (raw_title or "").lower()
    has_vneck = "v neck" in rt or "v-neck" in rt
    has_collared = "collar" in rt or "collared" in rt
    has_long_sleeve = "long sleeve" in rt or "long-sleeve" in rt

    parts = [color]

    if has_collared:
        parts.append("Collared")
    if has_vneck:
        parts.append("V Neck")
    if has_long_sleeve:
        parts.append("Long Sleeve")

    parts.append(product_type)

    title = " ".join(parts)
    title = enforce_title_rules(title)

    # final guard
    if not title:
        title = enforce_title_rules(f"{color} {product_type}")

    return title


def build_description(title: str) -> str:
    """
    1–3 sentences, title appears in first sentence, fun/friendly tone, SEO-friendly.
    """
    # Keep it short and clean
    s1 = f"{title} is a fun easy way to look instantly put together"
    s2 = "The sleek fitted silhouette layers smoothly and styles easily from day to night"
    return enforce_sentence_rules(f"{s1}. {s2}.")


def enforce_sentence_rules(text: str) -> str:
    # no special rules requested beyond title; keep readable
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_details(product_type: str, color: str) -> List[str]:
    """
    Plain text bullets, preceded by 'Details:' in UI, last bullet must be 'Imported.'
    """
    color = standardize_color(color) or "Black"
    # For the specific bodysuit case you described
    if product_type == "Bodysuit Top":
        bullets = [
            "Solid color bodysuit top with long sleeves",
            "Contrast white collar and cuffs for a dressed up look",
            "V neck collared neckline style",
            "Fitted silhouette for clean layering under jackets and skirts",
            "Great for office nights out and casual chic outfits",
            "Imported.",
        ]
        return bullets

    # Generic fallback
    return [
        f"Solid color {product_type.lower()}",
        "Easy styling for day to night looks",
        "Comfortable fit for everyday wear",
        "Imported.",
    ]


def generate_hidden_tags(product_type: str, color: str, raw_title: str) -> List[str]:
    """
    5 tags, no vendor codes, no forbidden words, lowercase.
    """
    color = (standardize_color(color) or "Black").lower()
    base = {
        "bodysuit top": ["bodysuit", "long sleeve", "collared", "v neck", "layering"],
        "romper": ["romper", "tie waist", "lightweight", "vacation", "easy outfit"],
        "dress": ["dress", "day to night", "easy styling", "wardrobe staple", "comfortable"],
        "top": ["top", "easy styling", "wardrobe staple", "layering", "everyday"],
    }

    key = product_type.lower()
    tags = base.get(key, base["top"]).copy()

    # include color if it’s a real color
    if color and color in COLOR_CANONICAL:
        tags.insert(0, color)

    # clean
    cleaned: List[str] = []
    for t in tags:
        t2 = strip_vendor_codes(t)
        t2 = remove_forbidden_words(t2)
        t2 = TITLE_ALLOWED_RE.sub(" ", t2).strip().lower()
        t2 = re.sub(r"\s+", " ", t2)
        if t2 and t2 not in cleaned:
            cleaned.append(t2)

    return cleaned[:5]


def build_listing(api_raw_title: str, vendor_color: str, variants_text: str, size_chart_json: Dict[str, Any], img: Optional[Image.Image]) -> Dict[str, Any]:
    # Vision color (fallback when vendor color is a code)
    vision_color, vision_conf = ("", "Low")
    if img is not None:
        vision_color, vision_conf = vision_extract_color(img)

    # Choose final color
    final_color = ""
    if vendor_color and not looks_like_vendor_code(vendor_color):
        final_color = standardize_color(vendor_color)
    else:
        final_color = standardize_color(vision_color) or "Black"

    product_type = infer_product_type(api_raw_title)

    title = build_clean_title(product_type, final_color, api_raw_title)
    desc = build_description(title)
    details = build_details(product_type, final_color)
    tags = generate_hidden_tags(product_type, final_color, api_raw_title)

    # Standardize variants color
    variants = parse_variants(variants_text)
    standardized_variants = []
    for v in variants:
        standardized_variants.append(
            {
                "Color": final_color,
                "Size": v.get("size", "").upper(),
            }
        )

    # Clean size chart
    size_table_md, size_warnings = clean_size_chart(size_chart_json)

    # Validators
    errors = []
    if len(title) > TITLE_MAX_CHARS:
        errors.append(f"Title too long: {len(title)} chars (max {TITLE_MAX_CHARS}).")
    if "-" in title or any(ch in title for ch in r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""):
        errors.append("Title contains punctuation or hyphens.")
    if any(w in title.lower().split() for w in FORBIDDEN_WORDS):
        errors.append("Title contains forbidden word (women/female/etc).")
    if looks_like_vendor_code(title):
        errors.append("Title contains vendor/style code.")
    if len(tags) != 5:
        errors.append("Hidden tags must be exactly 5.")

    return {
        "final_color": final_color,
        "vision_confidence": vision_conf,
        "title": title,
        "description": desc,
        "details": details,
        "hidden_tags": tags,
        "standardized_variants": standardized_variants,
        "size_table_md": size_table_md,
        "size_warnings": size_warnings,
        "errors": errors,
    }


def bullets_plain_text(bullets: List[str]) -> str:
    # plain text bullets (no HTML)
    return "\n".join([f"• {b}" for b in bullets])


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="AI Product Listing Prototype", layout="wide")
st.title("AI Product Listing Demo (Image + Simulated API)")
st.caption("Simulates API import + Vision + Copywriting with strict validators.")

# Initialize state once
if "api_raw_title" not in st.session_state:
    st.session_state["api_raw_title"] = ""
if "api_vendor_color" not in st.session_state:
    st.session_state["api_vendor_color"] = ""
if "api_variants_text" not in st.session_state:
    st.session_state["api_variants_text"] = ""
if "api_size_chart_json" not in st.session_state:
    st.session_state["api_size_chart_json"] = json.dumps(DEMO_API_PAYLOAD["size_chart_json"], indent=2)
if "final_output" not in st.session_state:
    st.session_state["final_output"] = None

# --- Vendor URL entry (single input, no duplicates) ---
ali_url = st.text_input("Vendor URL (demo)", value=DEMO_VENDOR_URL, key="ali_url_input")
st.markdown(f"[Open vendor page]({ali_url})")

# One button only (no “Use demo URL”)
if st.button("Load from URL (simulated API)", key="load_sim_api"):
    # Load simulated payload into session state (reliable demo)
    st.session_state["api_raw_title"] = DEMO_API_PAYLOAD["raw_title"]
    st.session_state["api_vendor_color"] = DEMO_API_PAYLOAD["vendor_color"]
    st.session_state["api_variants_text"] = DEMO_API_PAYLOAD["variants_text"]
    st.session_state["api_size_chart_json"] = json.dumps(DEMO_API_PAYLOAD["size_chart_json"], indent=2)
    st.session_state["final_output"] = None
    st.rerun()

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Simulated API Input")

    api_raw_title = st.text_input(
        "Raw Title (from vendor API)",
        value=st.session_state["api_raw_title"] or DEMO_API_PAYLOAD["raw_title"],
        key="api_raw_title_input",
    )

    api_vendor_color = st.text_input(
        "Vendor Color (often messy)",
        value=st.session_state["api_vendor_color"] or DEMO_API_PAYLOAD["vendor_color"],
        key="api_vendor_color_input",
        help="If this looks like a style code, we ignore it and use vision color instead.",
    )

    api_variants_text = st.text_area(
        "Variants (raw text)",
        value=st.session_state["api_variants_text"] or DEMO_API_PAYLOAD["variants_text"],
        height=120,
        key="api_variants_text_input",
    )

    api_size_chart_json_text = st.text_area(
        "Size Chart JSON (from API)",
        value=st.session_state["api_size_chart_json"],
        height=220,
        key="api_size_chart_json_input",
    )

with right:
    st.subheader("Product Image (for vision)")
    uploaded = st.file_uploader("Upload an image (png/jpg)", type=["png", "jpg", "jpeg"], key="img_uploader")
    img: Optional[Image.Image] = None
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image (used for color inference)", use_container_width=True)
    else:
        st.info("Optional: upload an image so the demo can infer and standardize the color name.")

st.divider()

gen = st.button("Generate Clean Listing", type="primary", key="generate_btn")

if gen:
    # Parse size chart json safely
    size_chart_obj: Dict[str, Any] = {}
    try:
        size_chart_obj = json.loads(api_size_chart_json_text)
        if not isinstance(size_chart_obj, dict):
            size_chart_obj = {}
    except Exception:
        size_chart_obj = {}

    out = build_listing(
        api_raw_title=api_raw_title,
        vendor_color=api_vendor_color,
        variants_text=api_variants_text,
        size_chart_json=size_chart_obj,
        img=img,
    )

    st.session_state["final_output"] = out

out = st.session_state.get("final_output")

if out is None:
    st.info("Click **Generate Clean Listing** to produce Product Title, Description, Hidden Tags, and cleaned Size Chart.")
else:
    # --- Output section ---
    st.subheader("Output (Paste Ready)")

    if out["errors"]:
        st.error("Validator issues: " + " | ".join(out["errors"]))
    else:
        st.success("All checks passed (title rules, format, tags, no style codes).")

    st.markdown("**Standardized Color**")
    st.code(out["final_color"], language="text")
    st.caption(f"Vision confidence: {out['vision_confidence']}")

    st.markdown("**Product Title (≤ 60 chars, no punctuation, no hyphens)**")
    st.code(out["title"], language="text")
    st.caption(f"Length: {len(out['title'])} characters")

    st.markdown("**Product Description (1–3 sentences)**")
    st.code(out["description"], language="text")

    st.markdown("**Details:**")
    st.code(bullets_plain_text(out["details"]), language="text")

    st.markdown("**Hidden Tags (5)**")
    st.code(", ".join(out["hidden_tags"]), language="text")

    st.markdown("**Standardized Variants (review)**")
    if out["standardized_variants"]:
        st.json(out["standardized_variants"])
    else:
        st.info("No variants parsed from the raw variants text.")

    st.markdown("**Cleaned Size Chart**")
    if out["size_table_md"]:
        st.markdown(out["size_table_md"])
    else:
        st.warning("No size chart output.")

    if out["size_warnings"]:
        st.warning("Size chart: " + " | ".join(out["size_warnings"]))

    with st.expander("AI Facts JSON (what the AI used)"):
        st.json(out)
