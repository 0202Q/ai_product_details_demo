import json
import re
from dataclasses import dataclass
from html import escape
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image


# =============================
# Demo constants (simulated API)
# =============================

DEMO_VENDOR_URL = "https://www.aliexpress.us/item/3256810240290544.html"

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
    # Simulated "size chart via API" (what you'd ideally get from AE API, seller feed, or enrichment service)
    "size_chart_json": {
        "unit": "cm",
        "columns": ["Size", "Bust", "Shoulder", "Sleeve", "Waist", "Hip", "Length"],
        "rows": [
            ["S", 80, 36, 60, "64.5-98", 73, 71],
            ["M", 84, 37, 61, "68.5-102", 77, 72.4],
            ["L", 90, 38, 62, "74.5-108", 83, 74.6],
            ["XL", 96, 39, 63, "80.5-114", 89, 76.8],
        ],
        "notes": ["Allow 2–3cm differences due to manual measurement."],
    },
}


# =============================
# Rules / helpers
# =============================

STYLE_CODE_RE = re.compile(r"\b[A-Z]{2,6}\d{2,6}[-_]?\d*\b")  # e.g., BKK1196-1
YEAR_RE = re.compile(r"\b(20\d{2})\b")
PUNCT_RE = re.compile(r"[^\w\s]")  # remove punctuation
MULTISPACE_RE = re.compile(r"\s+")


def strip_style_codes(text: str) -> str:
    if not text:
        return ""
    return STYLE_CODE_RE.sub("", text).strip()


def normalize_spaces(text: str) -> str:
    return MULTISPACE_RE.sub(" ", (text or "").strip())


def remove_forbidden_words(text: str, forbidden: List[str]) -> str:
    t = f" {text} "
    for w in forbidden:
        t = re.sub(rf"\b{re.escape(w)}\b", " ", t, flags=re.IGNORECASE)
    return normalize_spaces(t)


def remove_punctuation_and_hyphens(text: str) -> str:
    # Title rule: no punctuation, no hyphens
    # Hyphens are punctuation too, but keep explicit rule.
    t = (text or "").replace("-", " ")
    t = PUNCT_RE.sub(" ", t)
    return normalize_spaces(t)


def enforce_title_limit(title: str, max_chars: int = 60) -> str:
    t = normalize_spaces(title)
    if len(t) <= max_chars:
        return t
    # Trim by words, keep under limit
    words = t.split()
    out = ""
    for w in words:
        candidate = (out + " " + w).strip()
        if len(candidate) > max_chars:
            break
        out = candidate
    return out if out else t[:max_chars].strip()


def standardize_color(color: str) -> str:
    """
    Convert vendor-ish color strings into standard customer-friendly names.
    If it's a style code, return empty and rely on Vision.
    """
    c = normalize_spaces(strip_style_codes(color)).lower()

    if not c:
        return ""

    # common mappings
    mapping = {
        "black": "Black",
        "white": "White",
        "ivory": "Ivory",
        "cream": "Cream",
        "beige": "Beige",
        "tan": "Tan",
        "brown": "Brown",
        "camel": "Camel",
        "navy": "Navy",
        "blue": "Blue",
        "light blue": "Light Blue",
        "dark blue": "Dark Blue",
        "green": "Green",
        "light green": "Light Green",
        "sage": "Sage",
        "sage green": "Sage Green",
        "olive": "Olive",
        "olive green": "Olive Green",
        "pink": "Pink",
        "red": "Red",
        "burgundy": "Burgundy",
        "gray": "Gray",
        "grey": "Gray",
        "purple": "Purple",
        "yellow": "Yellow",
        "orange": "Orange",
    }

    # exact match first
    if c in mapping:
        return mapping[c]

    # fuzzy contains
    for k, v in mapping.items():
        if k in c:
            return v

    # fallback: title-case cleaned token
    return c.title()


def parse_variants(variants_text: str) -> List[Dict[str, str]]:
    """
    Parse lines like: "Color: BKK1196-1, Size: M"
    """
    variants = []
    for line in (variants_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # simple parse
        m_color = re.search(r"color\s*:\s*([^,]+)", line, flags=re.I)
        m_size = re.search(r"size\s*:\s*([A-Za-z0-9]+)", line, flags=re.I)
        variants.append(
            {
                "raw": line,
                "color": normalize_spaces(m_color.group(1)) if m_color else "",
                "size": normalize_spaces(m_size.group(1)) if m_size else "",
            }
        )
    return variants


def clean_size_chart(chart_json: dict) -> Tuple[str, List[str]]:
    """
    Returns a markdown table + warnings list.
    """
    warnings = []
    if not chart_json:
        return "", ["No size chart found."]

    unit = chart_json.get("unit", "")
    cols = chart_json.get("columns") or []
    rows = chart_json.get("rows") or []
    notes = chart_json.get("notes") or []

    if not cols or not rows:
        return "", ["Size chart payload missing columns/rows."]

    # Build markdown table
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for r in rows:
        r_str = [str(x) for x in r]
        # pad/truncate to match cols
        if len(r_str) < len(cols):
            r_str += [""] * (len(cols) - len(r_str))
        if len(r_str) > len(cols):
            r_str = r_str[: len(cols)]
        body_lines.append("| " + " | ".join(r_str) + " |")

    md = "\n".join([header, sep] + body_lines)

    if unit:
        warnings.append(f"Measurements are in {unit}.")
    for n in notes:
        warnings.append(str(n))

    return md, warnings


# =============================
# "AI" layers (mocked but realistic)
# =============================

def vision_extract_color(img: Optional[Image.Image]) -> Dict[str, str]:
    """
    Fake Vision AI: infer garment color from image darkness.
    - dark image -> Black
    - mid -> Navy/Charcoal
    - bright -> White/Cream
    This is intentionally simple for demo reliability.
    """
    if img is None:
        return {"color_name": "", "confidence": "Low", "evidence": "No image provided."}

    # downscale for speed
    small = img.convert("RGB").resize((64, 64))
    pixels = list(small.getdata())
    # average brightness
    avg = sum((r + g + b) / 3 for r, g, b in pixels) / len(pixels)

    if avg < 70:
        return {"color_name": "Black", "confidence": "High", "evidence": "Garment appears very dark overall."}
    if avg < 110:
        return {"color_name": "Charcoal", "confidence": "Medium", "evidence": "Garment appears dark gray/blue overall."}
    if avg < 170:
        return {"color_name": "Navy", "confidence": "Low", "evidence": "Garment appears mid-tone; could be navy or gray."}
    return {"color_name": "White", "confidence": "Low", "evidence": "Garment appears very light overall."}


def extract_facts_from_api(raw_title: str, vendor_color: str, variants_text: str) -> Dict[str, str]:
    """
    Extract core facts from the simulated API payload.
    """
    title = normalize_spaces(raw_title)
    title = remove_forbidden_words(title, ["women", "woman", "female", "ladies", "girls"])
    title = strip_style_codes(title)
    title = normalize_spaces(title)

    # Detect garment type keywords (minimal for one-category demo)
    lt = title.lower()
    garment = "bodysuit" if "bodysuit" in lt else "top"

    sleeve = "long sleeve" if "long sleeve" in lt else ""
    neckline = "v neck" if "v neck" in lt or "v-neck" in lt else ""

    # Vendor color might be a style code; standardize_color will drop it.
    vendor_color_std = standardize_color(vendor_color)

    variants = parse_variants(variants_text)

    return {
        "raw_title_cleaned": title,
        "garment": garment,
        "sleeve": sleeve,
        "neckline": neckline,
        "vendor_color": vendor_color_std,  # may be ""
        "variants_count": str(len(variants)),
    }


def choose_final_color(vendor_color_std: str, vision_color: str) -> str:
    # Use vendor color only if it looks real (not empty after standardize).
    c = vendor_color_std.strip()
    if c:
        return c
    return vision_color.strip() or "Black"  # safe fallback for demo


def generate_hidden_tags(garment: str, color: str, sleeve: str, neckline: str) -> List[str]:
    """
    Exactly 5 tags. No style codes.
    """
    base = []
    if garment:
        base.append(garment.lower())
    if color:
        base.append(color.lower())
    if sleeve:
        base.append(sleeve.lower().replace(" ", "_"))
    if neckline:
        base.append(neckline.lower().replace(" ", "_"))

    # fill to 5 with sensible defaults for this category
    defaults = ["one_piece", "layering", "going_out", "minimal"]
    for d in defaults:
        if len(base) >= 5:
            break
        base.append(d)

    # unique, preserve order, ensure 5
    seen = set()
    out = []
    for t in base:
        t = strip_style_codes(t)
        t = re.sub(r"[^\w]+", "_", t).strip("_")
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) == 5:
            break

    # if still short, pad
    while len(out) < 5:
        out.append(f"tag_{len(out)+1}")
    return out


def generate_listing_copy(
    raw_title_cleaned: str,
    final_color: str,
    garment: str,
    sleeve: str,
    neckline: str,
) -> Dict[str, str]:
    """
    Mock LLM output but enforced by validators:
    - title <= 60 chars
    - no punctuation / hyphens
    - removes women/female/etc
    - no style codes
    - fun + friendly tone, SEO-ish
    """
    # Build a clean title candidate
    # Keep it simple + consistent: Color + key descriptors + garment
    # Also: avoid repeating "long sleeve" twice.
    title_parts = []

    if final_color:
        title_parts.append(final_color)

    # Collared / V neck cues for this example
    # If vendor title doesn't include "collared", infer from image is hard;
    # but your example wants collared v neck long sleeve bodysuit.
    # We'll derive "Collared" if "collar" appears in raw_title_cleaned.
    lt = raw_title_cleaned.lower()
    if "collar" in lt or "collared" in lt:
        title_parts.append("Collared")

    if neckline:
        title_parts.append("V Neck")

    if sleeve:
        title_parts.append("Long Sleeve")

    # garment label
    if garment == "bodysuit":
        title_parts.append("Bodysuit Top")
    else:
        title_parts.append("Top")

    title = " ".join(title_parts)
    title = strip_style_codes(title)
    title = remove_forbidden_words(title, ["women", "woman", "female", "ladies", "girls"])
    title = remove_punctuation_and_hyphens(title)
    title = enforce_title_limit(title, 60)

    # Description (1–3 sentences). Must include title in first sentence.
    desc = (
        f"{title} is a fun easy way to look instantly put together. "
        f"The solid {final_color.lower()} silhouette is sleek and fitted for easy day to night styling. "
        "Style it with denim for casual plans or dress it up with trousers and heels."
    )
    # Keep to 1–3 sentences
    desc = " ".join(desc.split())  # normalize
    # If you want 2 sentences max, you can trim here; keeping 3 is allowed.

    # Details bullets (plain text bullets) + ending Imported.
    details = [
        f"Solid color {garment} top with long sleeves" if garment else "Solid color top with long sleeves",
        "Contrast collar and cuffs for a dressed up look",
        "V neck collared neckline style",
        "Fitted silhouette for clean layering under jackets and skirts",
        "Great for office nights out and casual chic outfits",
        "Imported.",
    ]

    # Ensure no punctuation/hyphens in title already; details can have punctuation in sentences,
    # but user asked "plain text with bullets" (we comply).
    return {
        "title": title,
        "description": desc,
        "details_bullets": "\n".join([f"• {d}" for d in details]),
    }


def validate_title_rules(title: str) -> List[str]:
    errs = []
    if len(title) > 60:
        errs.append(f"Title length is {len(title)} (must be ≤ 60).")
    if "-" in title:
        errs.append("Title contains hyphen (not allowed).")
    if re.search(r"[^\w\s]", title):
        errs.append("Title contains punctuation (not allowed).")
    if re.search(r"\bwomen\b", title, flags=re.I):
        errs.append('Title contains "women" (must be removed).')
    if STYLE_CODE_RE.search(title):
        errs.append("Title contains a style code (not allowed).")
    return errs


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="AI Product Listing Prototype", layout="wide")
st.title("AI Product Listing Demo (Image + Simulated API)")
st.caption("Simulates API import + Vision + Copywriting with strict validators.")

# --- Demo URL entry (top of page) ---
ali_url = st.text_input("Vendor URL (demo)", value=DEMO_VENDOR_URL, key="ali_url_input")
st.markdown(f"[Open vendor page]({ali_url})")

# --- Demo URL entry (top of page) ---
ali_url = st.text_input(
    "Vendor URL (demo)",
    value=DEMO_VENDOR_URL,
    key="ali_url_input",
)
st.markdown(f"[Open vendor page]({ali_url})")

# Only one button now
if st.button("Load from URL (simulated API)", key="load_sim_api"):
    st.session_state["api_raw_title"] = DEMO_API_PAYLOAD["raw_title"]
    st.session_state["api_vendor_color"] = DEMO_API_PAYLOAD["vendor_color"]
    st.session_state["api_variants_text"] = DEMO_API_PAYLOAD["variants_text"]
    st.session_state["api_size_chart_json"] = json.dumps(
        DEMO_API_PAYLOAD["size_chart_json"], indent=2
    )
    st.rerun()

if c2.button("Load from URL (simulated API)"):
    st.session_state["ali_url_input"] = DEMO_VENDOR_URL

    st.session_state["api_raw_title"] = DEMO_API_PAYLOAD["raw_title"]
    st.session_state["api_vendor_color"] = DEMO_API_PAYLOAD["vendor_color"]
    st.session_state["api_variants_text"] = DEMO_API_PAYLOAD["variants_text"]
    st.session_state["api_size_chart_json"] = json.dumps(
        DEMO_API_PAYLOAD["size_chart_json"], indent=2
    )

    st.rerun()
    # Set the simulated API payload into session state (reliable demo, no scraping)
    st.session_state["api_raw_title"] = DEMO_API_PAYLOAD["raw_title"]
    st.session_state["api_vendor_color"] = DEMO_API_PAYLOAD["vendor_color"]
    st.session_state["api_variants_text"] = DEMO_API_PAYLOAD["variants_text"]
    st.session_state["api_size_chart_json"] = json.dumps(DEMO_API_PAYLOAD["size_chart_json"], indent=2)
    st.rerun()

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Fake API Input")

    api_raw_title = st.text_input(
        "Raw Title (from vendor API)",
        value=st.session_state.get("api_raw_title", DEMO_API_PAYLOAD["raw_title"]),
        key="api_raw_title_input",
    )

    api_vendor_color = st.text_input(
        "Vendor Color (from API)",
        value=st.session_state.get("api_vendor_color", DEMO_API_PAYLOAD["vendor_color"]),
        key="api_vendor_color_input",
    )

    api_variants_text = st.text_area(
        "Raw Variants (from API)",
        value=st.session_state.get("api_variants_text", DEMO_API_PAYLOAD["variants_text"]),
        height=120,
        key="api_variants_text_input",
    )

    api_size_chart_json_str = st.text_area(
        "Size Chart (from API as JSON)",
        value=st.session_state.get("api_size_chart_json", json.dumps(DEMO_API_PAYLOAD["size_chart_json"], indent=2)),
        height=220,
        key="api_size_chart_json_input",
    )

    st.subheader("Image Input (Vision)")
    uploaded = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"], key="img_upload")
    img = None
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)

    st.subheader("Category (single-category demo)")
    st.info("Demo uses one category only: Apparel & Accessories > Bodysuits")
    category = "Apparel & Accessories > Bodysuits"

    gen = st.button("Generate Clean Listing", type="primary", key="gen_btn")

with right:
    st.subheader("Output (Paste Ready)")

    if not st.session_state.get("final_output"):
        st.info("Load the simulated API payload and upload an image, then click Generate.")
    else:
        out = st.session_state["final_output"]
        st.success("Generated")

        st.markdown("### Product Title (≤ 60 chars)")
        st.code(out["title"], language="text")

        st.markdown("### Product Description (1–3 sentences)")
        st.write(out["description"])

        st.markdown("### Details:")
        st.code(out["details_bullets"], language="text")

        st.markdown("### Hidden Tags (5)")
        st.code(", ".join(out["hidden_tags"]), language="text")

        st.markdown("### Size Guide (cleaned)")
        if out["size_table_md"]:
            st.markdown(out["size_table_md"])
        if out["size_warnings"]:
            st.warning("Size chart notes: " + " | ".join(out["size_warnings"]))

        st.markdown("### Standardized Color")
        st.code(out["final_color"], language="text")

        st.markdown("### Validators")
        if out["errors"]:
            st.error(" | ".join(out["errors"]))
        else:
            st.success("All checks passed (title rules, tags, no style codes).")

        with st.expander("AI Facts JSON (what the AI used)"):
            st.json(out["facts"])


# =============================
# Generate action
# =============================

if gen:
    # 1) Extract facts from API
    api_facts = extract_facts_from_api(api_raw_title, api_vendor_color, api_variants_text)

    # 2) Vision color
    vision = vision_extract_color(img)
    vision_color = standardize_color(vision.get("color_name", ""))

    # 3) Final color decision
    final_color = choose_final_color(api_facts.get("vendor_color", ""), vision_color)

    # 4) Size chart cleaning (from API JSON)
    size_table_md = ""
    size_warnings: List[str] = []
    try:
        size_obj = json.loads(api_size_chart_json_str) if api_size_chart_json_str else {}
        size_table_md, size_warnings = clean_size_chart(size_obj)
    except Exception:
        size_table_md = ""
        size_warnings = ["Size chart JSON could not be parsed."]

    # 5) Generate listing copy (mock LLM)
    listing = generate_listing_copy(
        raw_title_cleaned=api_facts.get("raw_title_cleaned", ""),
        final_color=final_color,
        garment=api_facts.get("garment", "top"),
        sleeve=api_facts.get("sleeve", ""),
        neckline=api_facts.get("neckline", ""),
    )

    # 6) Hidden tags (5)
    hidden_tags = generate_hidden_tags(
        garment=api_facts.get("garment", ""),
        color=final_color,
        sleeve=api_facts.get("sleeve", ""),
        neckline=api_facts.get("neckline", ""),
    )

    # 7) Validate title rules
    errors = validate_title_rules(listing["title"])

    # Also validate tags: no style codes
    for t in hidden_tags:
        if STYLE_CODE_RE.search(t):
            errors.append("A hidden tag contains a style code (not allowed).")
            break

    # 8) Build final output
    final = {
        "category": category,
        "title": listing["title"],
        "description": listing["description"],
        "details_bullets": listing["details_bullets"],
        "hidden_tags": hidden_tags,
        "final_color": final_color,
        "size_table_md": size_table_md,
        "size_warnings": size_warnings,
        "errors": errors,
        "facts": {
            "api_facts": api_facts,
            "vision": vision,
            "final_color": final_color,
            "vendor_url": st.session_state.get("ali_url", ""),
        },
    }

    st.session_state["final_output"] = final
    st.rerun()
