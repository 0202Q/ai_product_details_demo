# app.py
# AI Product Listing Prototype (Apparel One Piece)
# - Fake API input (title, variants, optional size chart payload)
# - Image analysis layer -> facts JSON
# - LLM-like generation layer -> listing
# - Hard validators enforce your formatting rules

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
MAX_TITLE_LEN = 60
BANNED_WORDS = {"women", "womens", "woman",
                "female", "girls", "girl", "ladies", "lady"}

COLOR_MAP = {
    "light green": "Sage Green",
    "sage green": "Sage Green",
    "sage": "Sage Green",
    "army green": "Olive Green",
    "olive green": "Olive Green",
    "dark green": "Forest Green",
    "forest green": "Forest Green",
    "black": "Black",
    "white": "White",
    "green": "Green",
}

SIZE_ORDER = ["XS", "S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL"]

TITLE_ALLOWED_RE = re.compile(r"[^A-Za-z0-9\s]")


# -----------------------------
# STYLE CODE STRIPPING (BKK1196-1 etc)
# -----------------------------
def looks_like_style_code(token: str) -> bool:
    if not token:
        return False
    t = token.strip()
    t2 = re.sub(r"[^A-Za-z0-9]", "", t)
    if len(t2) < 5:
        return False
    has_alpha = bool(re.search(r"[A-Za-z]", t2))
    has_digit = bool(re.search(r"\d", t2))
    return (has_alpha and has_digit) or (t2.isupper() and has_digit and len(t2) >= 5)


def strip_style_codes(text: str) -> str:
    if not text:
        return ""
    tokens = re.split(r"\s+", text.strip())
    kept = []
    for tok in tokens:
        clean_tok = tok.strip(",;:()[]{}")
        if looks_like_style_code(clean_tok):
            continue
        kept.append(tok)
    return " ".join(kept).strip()


# -----------------------------
# TITLE VALIDATION (your rules)
# -----------------------------
def sanitize_title(raw_title: str, max_len: int = MAX_TITLE_LEN) -> str:
    if not raw_title:
        return ""
    raw_title = strip_style_codes(raw_title)

    # remove punctuation/hyphens etc
    t = TITLE_ALLOWED_RE.sub("", raw_title)
    t = re.sub(r"\s{2,}", " ", t).strip()

    # remove banned words + style codes again
    tokens = []
    for w in t.split():
        wl = w.lower()
        if wl in BANNED_WORDS:
            continue
        if looks_like_style_code(w):
            continue
        tokens.append(w)
    t = " ".join(tokens).strip()

    # trim to <= 60 without cutting word if possible
    if len(t) <= max_len:
        return t
    cut = t[:max_len].rstrip()
    if " " in cut:
        cut2 = cut[:cut.rfind(" ")].rstrip()
        if len(cut2) >= 25:
            return cut2
    return cut


def no_punctuation_or_hyphen(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9 ]+", s))


# -----------------------------
# VARIANT + SIZE HELPERS
# -----------------------------
def extract_sizes(text: str) -> List[str]:
    if not text:
        return []
    found = re.findall(r"\b(5XL|4XL|3XL|2XL|XL|L|M|S|XS)\b", text.upper())
    sset = {s for s in found}
    ordered = [s for s in SIZE_ORDER if s in sset]
    remainder = sorted([s for s in sset if s not in ordered])
    return ordered + remainder


def standardize_color(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if looks_like_style_code(raw):
        return ""  # treat as invalid color, fall back to vision
    key = raw.lower()
    return COLOR_MAP.get(key, raw.title())


def standardize_variant_colors(variants_text: str, fallback_color: str) -> str:
    if not variants_text:
        return ""

    fb = fallback_color.strip() if fallback_color else ""

    def repl(match: re.Match) -> str:
        raw_color = (match.group(1) or "").strip()
        std = standardize_color(raw_color)
        if not std and fb:
            std = fb
        if not std:
            std = "Unknown"
        return f"Color: {std}"

    updated = re.sub(r"(?i)Color:\s*([^,\n]+)", repl, variants_text)
    updated = strip_style_codes(updated)
    return updated


# -----------------------------
# SIZE CHART CLEANER (API payload)
# -----------------------------
def clean_size_chart_payload(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {"unit": "cm", "rows": [], "warnings": ["Missing size chart payload"]}

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            unit = data.get("unit", "cm")
            rows = data.get("rows", [])
            if isinstance(rows, list) and rows:
                return {"unit": unit, "rows": rows, "warnings": []}
            return {"unit": unit, "rows": [], "warnings": ["JSON parsed but rows were empty"]}
        except Exception:
            pass

    # CSV fallback
    rows = []
    for ln in [x.strip() for x in raw.splitlines() if x.strip()]:
        if "," not in ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            continue
        size = parts[0].upper()
        if size not in SIZE_ORDER:
            continue
        row = {"size": size}
        # size,bust,waist,hip,length
        if len(parts) >= 5:
            row["bust"] = parts[1]
            row["waist"] = parts[2]
            row["hip"] = parts[3]
            row["length"] = parts[4]
        rows.append(row)

    if not rows:
        return {"unit": "cm", "rows": [], "warnings": ["Could not parse size chart payload"]}

    return {"unit": "cm", "rows": rows, "warnings": []}


def render_size_chart_table(size_chart: Dict[str, Any]) -> str:
    rows = size_chart.get("rows") or []
    unit = size_chart.get("unit", "cm")
    if not rows:
        return "Size chart not available."

    cols = ["size", "bust", "waist", "hip", "length"]
    lines = []
    lines.append(f"Size Guide ({unit})")
    lines.append(" | ".join([c.upper() for c in cols]))
    lines.append(" | ".join(["---"] * len(cols)))
    for r in rows:
        lines.append(" | ".join([str(r.get(c, "")) for c in cols]))
    return "\n".join(lines)


# -----------------------------
# VISION LAYER (image -> facts JSON)
# -----------------------------
def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def infer_main_color_from_image(img: Image.Image) -> Tuple[str, str]:
    """
    Demo vision: predicts Black/White confidently; Green moderately.
    """
    img = _to_rgb(img).resize((128, 128))
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr.mean(axis=(0, 1)).tolist()
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if brightness < 0.28:
        return "Black", "High"
    if brightness > 0.82:
        return "White", "High"
    if (g - r) > 0.08 and (g - b) > 0.05:
        return "Green", "Medium"
    return "", "Low"


def detect_contrast_collar_cuffs(img: Image.Image) -> bool:
    img = _to_rgb(img)
    w, h = img.size
    if w < 80 or h < 80:
        return False

    arr = np.asarray(img).astype(np.float32)
    bright = 0.2126 * arr[:, :, 0] + 0.7152 * \
        arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    torso = bright[int(h * 0.30):int(h * 0.75), int(w * 0.30):int(w * 0.70)]
    neck = bright[int(h * 0.05):int(h * 0.25), int(w * 0.35):int(w * 0.65)]
    cuff_l = bright[int(h * 0.65):int(h * 0.88), int(w * 0.05):int(w * 0.25)]
    cuff_r = bright[int(h * 0.65):int(h * 0.88), int(w * 0.75):int(w * 0.95)]

    torso_mean = float(np.mean(torso))
    neck_mean = float(np.mean(neck))
    cuff_mean = float((np.mean(cuff_l) + np.mean(cuff_r)) / 2.0)

    return torso_mean < 110 and (neck_mean - torso_mean) > 35 and (cuff_mean - torso_mean) > 35


def vision_extract_facts(img: Optional[Image.Image]) -> Dict[str, Any]:
    """
    This is your 'real AI vision' slot.
    In demo mode it uses heuristics, but returns a structured JSON like a vision model would.
    Swap this function later with a real Vision API call.
    """
    if img is None:
        return {
            "color": "",
            "color_confidence": "Low",
            "contrast_collar_cuffs": False,
            "evidence": "No image provided"
        }

    color, conf = infer_main_color_from_image(img)
    contrast = detect_contrast_collar_cuffs(img)

    evidence = []
    if color:
        evidence.append(f"Main garment appears {color.lower()}")
    if contrast:
        evidence.append("Contrast collar and cuffs detected")
    if not evidence:
        evidence.append("Limited signal from image")

    return {
        "color": color,
        "color_confidence": conf,
        "contrast_collar_cuffs": contrast,
        "evidence": "; ".join(evidence)
    }


# -----------------------------
# "FAKE API" (title/specs/variants) -> facts
# -----------------------------
def infer_product_type(title: str) -> str:
    t = (title or "").lower()
    if "romper" in t:
        return "romper"
    if "bodysuit" in t:
        return "bodysuit"
    if "jumpsuit" in t:
        return "jumpsuit"
    if "dress" in t:
        return "dress"
    return "one piece"


def api_extract_facts(raw_title: str, vendor_color: str, variants: str) -> Dict[str, Any]:
    title = strip_style_codes(raw_title or "")
    product_type = infer_product_type(title)
    sizes = extract_sizes(title + "\n" + (variants or ""))

    # Title-based features (only if explicitly present)
    t = title.lower()
    long_sleeve = "long sleeve" in t
    v_neck = "v neck" in t or "v-neck" in t
    collared = "collar" in t or "collared" in t
    collared_v_neck = bool(v_neck and collared)

    return {
        "product_type": product_type,
        "sizes": sizes,
        "title_features": {
            "long_sleeve": long_sleeve,
            "v_neck": v_neck,
            "collared": collared,
            "collared_v_neck": collared_v_neck
        },
        "vendor_color": standardize_color(vendor_color or ""),
        "clean_title_seed": title
    }


# -----------------------------
# LLM GENERATION LAYER (facts JSON -> listing)
# For demo: deterministic "mock LLM" that behaves like a model, but never leaks codes.
# Swap this function later with a real LLM call.
# -----------------------------
def mock_llm_generate_listing(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates:
    - product_name (validated)
    - product_description (your format)
    - hidden_tags (5)
    """
    product_type = facts.get("product_type", "one piece")
    sizes = facts.get("sizes", [])
    tf = facts.get("title_features", {})
    vision = facts.get("vision", {})
    color = facts.get("final_color", "")

    # Build professional title parts (avoid duplicates)
    parts = []
    if color:
        parts.append(color)

    # If we know it's the bodysuit w collar/vneck/long sleeve, create the exact intended title shape
    if product_type == "bodysuit":
        if tf.get("collared_v_neck") or (tf.get("collared") and tf.get("v_neck")) or facts.get("force_collared_v_neck"):
            parts.append("Collared V Neck")
        if tf.get("long_sleeve") or facts.get("force_long_sleeve"):
            parts.append("Long Sleeve")
        parts.append("Bodysuit Top")
    elif product_type == "romper":
        parts.append("Tie Waist Romper")
    else:
        parts.append("One Piece Outfit")

    # De-dupe phrase blocks
    dedup = []
    seen = set()
    for p in parts:
        k = p.lower().strip()
        if k and k not in seen:
            dedup.append(p)
            seen.add(k)

    product_name = sanitize_title(" ".join(dedup), MAX_TITLE_LEN)

    # Details bullets
    # If bodysuit + black + contrast collar/cuffs, output EXACT bullet set you provided.
    contrast = bool(vision.get("contrast_collar_cuffs"))
    is_black = (color.lower() == "black")

    if product_type == "bodysuit" and is_black and contrast:
        details = [
            "Solid color bodysuit top with long sleeves",
            "Contrast white collar and cuffs for a dressed up look",
            "V neck collared neckline style",
            "Fitted silhouette for clean layering under jackets and skirts",
            "Great for office nights out and casual chic outfits",
        ]
    else:
        # Conservative fallback
        details = []
        if product_type == "romper":
            details.append("One piece romper style for easy outfits")
        if product_type == "bodysuit":
            details.append("Solid color bodysuit top")
        if tf.get("long_sleeve"):
            details.append("Long sleeve design")
        if contrast:
            details.append("Contrast collar and cuffs detail")
        if tf.get("collared_v_neck"):
            details.append("V neck collared neckline style")
        if color:
            details.append(f"Color {color}")
        if sizes:
            details.append(f"Sizes {', '.join(sizes)}")

    # Description: 1–3 sentences, first includes product name EXACTLY
    s1 = f"{product_name} is a fun easy way to look instantly put together."
    s2 = "Designed to style up fast while keeping your look clean and polished."
    description_lines = [f"{s1} {s2}", "", "Details:"]
    for b in details[:6]:
        description_lines.append(f"• {b}")
    description_lines.append("• Imported.")
    product_description = "\n".join(description_lines)

    # Hidden tags: exactly 5, never codes
    tags = []

    def add_tag(x: str):
        if not x:
            return
        t = re.sub(r"[^a-z0-9\s]", "", x.lower()).strip()
        t = re.sub(r"\s{2,}", " ", t)
        if t and not looks_like_style_code(t) and t not in tags:
            tags.append(t)

    add_tag(product_type if product_type != "one piece" else "one piece")
    if product_type == "bodysuit":
        add_tag("bodysuit")
    if tf.get("long_sleeve"):
        add_tag("long sleeve")
    if color:
        add_tag(color)
    # fill to 5
    for d in ["apparel", "one piece", "new arrival", "daily wear", "easy style", "casual"]:
        add_tag(d)
        if len(tags) == 5:
            break

    return {
        "product_name": product_name,
        "product_description": product_description,
        "hidden_tags": tags[:5],
    }


# -----------------------------
# FINAL VALIDATION (guarantees spec)
# -----------------------------
def validate_listing(output: Dict[str, Any]) -> List[str]:
    errs = []
    title = output.get("product_name", "")
    desc = output.get("product_description", "")
    tags = output.get("hidden_tags", [])

    if not title:
        errs.append("Missing product name")
    if len(title) > MAX_TITLE_LEN:
        errs.append("Title exceeds 60 characters")
    if not no_punctuation_or_hyphen(title):
        errs.append("Title contains punctuation or hyphens")
    if any(w in title.lower().split() for w in BANNED_WORDS):
        errs.append("Title contains banned gender words")
    if re.search(r"\b[B-Z]{2,}\d+\b", title):
        errs.append("Title may contain a style code")
    if not desc.startswith(title):
        errs.append(
            "Description first sentence must include exact product title")
    if "Details:" not in desc:
        errs.append("Description must include Details: heading")
    if not desc.strip().endswith("Imported."):
        errs.append("Description must end with Imported.")
    if not isinstance(tags, list) or len(tags) != 5:
        errs.append("Hidden tags must be exactly 5")
    for t in tags:
        if looks_like_style_code(t):
            errs.append("Hidden tags contain a style code")
            break
    return errs


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Product Listing Prototype", layout="wide")
st.title("AI Product Listing Demo (Image + Fake API)")
st.caption(
    "This prototype simulates API import + AI vision + AI copywriting with strict validators.")
# ---- Demo URL entry (top of page) ----


DEMO_VENDOR_URL = "https://www.aliexpress.us/item/3256810240290544.html"

ali_url = st.text_input("Vendor URL (demo)", value=DEMO_VENDOR_URL, key="ali_url")
st.markdown(f"[Open vendor page]({ali_url})")

c1, c2 = st.columns(2)
if c1.button("Use demo URL"):
    st.session_state["ali_url"] = DEMO_VENDOR_URL
    st.rerun()

if c2.button("Load from URL (simulated API)"):
    st.session_state["api_title"] = "Solid Color Slim 2025 Autumn Elegant Trend Tight Long Sleeve Bodysuit Tops"
    st.session_state["api_vendor_color"] = "BKK1196-1"
    st.session_state["api_variants"] = (
        "Color: BKK1196-1, Size: S\n"
        "Color: BKK1196-1, Size: M\n"
        "Color: BKK1196-1, Size: L\n"
        "Color: BKK1196-1, Size: XL"
    )
    st.rerun()
  
if ali_url:
    st.markdown(f"[Open vendor page]({ali_url})")

if st.button("Load from URL (simulated API)"):
    # Simulated API payload to make the demo reliable (no scraping dependency)
    st.session_state["api_title"] = "Solid Color Slim 2025 Autumn Elegant Trend Tight Long Sleeve Bodysuit Tops BKK1196-1"
    st.session_state["api_vendor_color"] = "BKK1196-1"
    st.session_state["api_variants"] = (
        "Color: BKK1196-1, Size: S\n"
        "Color: BKK1196-1, Size: M\n"
        "Color: BKK1196-1, Size: L\n"
        "Color: BKK1196-1, Size: XL"
    )
    st.session_state["api_size_payload"] = json.dumps({
        "unit": "cm",
        "rows": [
            {"size": "S", "bust": 80, "waist": "64.5-98", "hip": 73, "length": 71},
            {"size": "M", "bust": 84, "waist": "68.5-102", "hip": 77, "length": 72.4},
            {"size": "L", "bust": 90, "waist": "74.5-108", "hip": 83, "length": 74.6},
            {"size": "XL", "bust": 96, "waist": "80.5-114", "hip": 89, "length": 76.8},
        ]
    }, indent=2)
    st.rerun()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Fake API Input")
    if st.button("Load Bodysuit Example"):
        st.session_state["api_title"] = "Solid Color Slim 2025 Autumn Elegant Trend Tight Women's Long Sleeve Sexy Bodysuit Outfits Female Tops BKK1196-1"
        st.session_state["api_vendor_color"] = "BKK1196-1"
        st.session_state["api_variants"] = "Color: BKK1196-1, Size: S\nColor: BKK1196-1, Size: M\nColor: BKK1196-1, Size: L\nColor: BKK1196-1, Size: XL"
        st.session_state["api_size_payload"] = json.dumps({
            "unit": "cm",
            "rows": [
                {"size": "S", "bust": 80, "waist": "64.5-98", "hip": 73, "length": 71},
                {"size": "M", "bust": 84, "waist": "68.5-102",
                    "hip": 77, "length": 72.4},
                {"size": "L", "bust": 90, "waist": "74.5-108",
                    "hip": 83, "length": 74.6},
                {"size": "XL", "bust": 96, "waist": "80.5-114",
                    "hip": 89, "length": 76.8},
            ]
        }, indent=2)

    api_title = st.text_input(
        "API Product Title", value=st.session_state.get("api_title", ""))
    api_vendor_color = st.text_input(
        "API Vendor Color", value=st.session_state.get("api_vendor_color", ""))
    api_variants = st.text_area(
        "API Variants", value=st.session_state.get("api_variants", ""), height=120)
    api_size_payload = st.text_area("API Size Chart Payload (JSON or CSV)",
                                    value=st.session_state.get("api_size_payload", ""), height=160)

    st.subheader("Image Input")
    upload = st.file_uploader("Upload primary product image", type=[
                              "png", "jpg", "jpeg"])
    img0 = None
    if upload is not None:
        img0 = Image.open(upload)
        st.image(img0, caption="Primary Image", use_container_width=True)

    gen = st.button("Generate Professional Listing", type="primary")

with right:
    st.subheader("Output (Paste Ready)")
    if not st.session_state.get("final_output"):
        st.info("Generate to see Product Name, Description, Tags, and Size Guide.")


if gen:
    # Extract facts from fake API
    api_facts = api_extract_facts(api_title, api_vendor_color, api_variants)

    # Vision extraction from image
    vision = vision_extract_facts(img0)

    # Decide final color:
    # - use vendor color only if it's a real color
    # - otherwise use vision color
    final_color = api_facts.get("vendor_color", "")
    if not final_color:
        final_color = vision.get("color", "")
    # Standardize final color again just in case
    final_color = standardize_color(final_color)

    # Standardize variants with final color fallback (fixes Color: BKK1196-1 -> Color: Black)
    standardized_variants = standardize_variant_colors(
        api_variants, fallback_color=final_color)

    # Size chart
    size_chart = clean_size_chart_payload(api_size_payload)
    size_guide_text = render_size_chart_table(size_chart)

    # Compose full facts package (this is what you'd send to a real LLM)
    facts = {
        **api_facts,
        "vision": vision,
        "final_color": final_color,
        "standardized_variants": standardized_variants,
        # Optional: for demo if title doesn't mention collar/vneck but image shows it.
        # Keep off by default to avoid guessing too much.
        "force_collared_v_neck": False,
        "force_long_sleeve": api_facts.get("title_features", {}).get("long_sleeve", False),
    }

    # Generate listing (mock LLM)
    listing = mock_llm_generate_listing(facts)

    # Validate
    errors = validate_listing(listing)

    st.session_state["final_output"] = {
        "facts": facts,
        "listing": listing,
        "size_guide": size_guide_text,
        "errors": errors,
        "size_warnings": size_chart.get("warnings", []),
    }

out = st.session_state.get("final_output")
if out:
    with right:
        listing = out["listing"]

        st.markdown("### Product Name")
        st.code(listing["product_name"], language="text")
        st.caption(f"Length: {len(listing['product_name'])} / {MAX_TITLE_LEN}")

        st.markdown("### Product Description (paste into editor)")
        st.code(listing["product_description"], language="text")

        st.markdown("### Hidden Tags (5)")
        st.code(", ".join(listing["hidden_tags"]), language="text")

        st.markdown("### Sizing Guide (paste into Sizing Guide tab)")
        st.code(out["size_guide"], language="text")
        if out["size_warnings"]:
            st.warning("Size chart: " + "; ".join(out["size_warnings"]))

        st.markdown("### Standardized Variants (review)")
        st.code(out["facts"].get("standardized_variants", ""), language="text")

        if out["errors"]:
            st.error("Validator issues: " + " | ".join(out["errors"]))
        else:
            st.success(
                "All checks passed (title rules, format, tags, no style codes).")

        with st.expander("AI Facts JSON (what the AI used)"):
            st.json(out["facts"])
