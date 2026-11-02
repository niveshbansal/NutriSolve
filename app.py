"""
NutriSolver: Clinical Diet Optimization Co-Pilot
Hybrid neuro-symbolic app — Pandas hard-filters Indian foods, Gemini generates meals.
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

APP_DIR = Path(__file__).resolve().parent
FOODS_CSV = APP_DIR / "indian_foods.csv"

# Tried in order if list_models is empty or a listed id still 404s.
GEMINI_MODEL_FALLBACKS = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-2.5-flash",
    "gemini-flash-latest",
)

st.set_page_config(
    page_title="NutriSolver | Clinical Diet Co-Pilot",
    page_icon="🥗",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; }
      div[data-testid="stMetric"] {
        background: #f4faf6;
        border: 1px solid #d7eadc;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_gemini_client() -> genai.Client | None:
    """Build a Gemini client if GEMINI_API_KEY is present."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    return genai.Client(api_key=api_key)


def _normalize_model_id(name: str) -> str:
    """Strip the `models/` prefix used by list/get APIs."""
    name = (name or "").strip()
    if name.startswith("models/"):
        return name.split("/", 1)[1]
    return name


def _supports_generate_content(model: object) -> bool:
    """True if the model advertises generateContent (google-genai or legacy list_models)."""
    methods = (
        getattr(model, "supported_actions", None)
        or getattr(model, "supported_generation_methods", None)
        or []
    )
    if methods:
        return any(
            "generatecontent" in str(method).lower().replace("_", "")
            for method in methods
        )
    nid = _normalize_model_id(str(getattr(model, "name", ""))).lower()
    if not nid.startswith("gemini"):
        return False
    return not any(skip in nid for skip in ("embed", "imagen", "tts", "aqa"))


def _is_model_not_found(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in ("404", "not_found", "not found", "is not found", "not supported")
    )


def list_generate_content_models(client: genai.Client) -> list[str]:
    """
    Discover models that support generateContent.
    Uses google-genai `client.models.list()` (equivalent of legacy `genai.list_models()`).
    """
    found: list[str] = []

    try:
        for model in client.models.list():
            if _supports_generate_content(model):
                mid = _normalize_model_id(str(getattr(model, "name", "")))
                if mid:
                    found.append(mid)
    except Exception:
        pass

    # Legacy google.generativeai SDK, if it happens to be installed.
    try:
        import google.generativeai as legacy_genai  # type: ignore

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            legacy_genai.configure(api_key=api_key)
        for model in legacy_genai.list_models():
            if _supports_generate_content(model):
                mid = _normalize_model_id(str(getattr(model, "name", "")))
                if mid:
                    found.append(mid)
    except Exception:
        pass

    def _rank(name: str) -> tuple[int, str]:
        lower = name.lower()
        if "flash" in lower:
            return (0, lower)
        if "pro" in lower:
            return (1, lower)
        return (2, lower)

    unique: list[str] = []
    seen: set[str] = set()
    for name in sorted(found, key=_rank) + list(GEMINI_MODEL_FALLBACKS):
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def generate_content_with_fallback(
    client: genai.Client,
    contents,
    config: types.GenerateContentConfig | None = None,
):
    """
    Call generate_content, trying listed Flash/Pro models until one works.
    google-genai has no GenerativeModel class; this loop is the same fallback idea.
    """
    candidates = list_generate_content_models(client)
    cached = st.session_state.get("gemini_model")
    if cached:
        candidates = [cached] + [name for name in candidates if name != cached]

    errors: list[str] = []
    getter = getattr(client.models, "get", None)

    for name in candidates:
        try:
            # Drop ids that 404 before spending a generate call (when get() exists).
            if getter is not None:
                try:
                    getter(model=name)
                except Exception as probe_exc:
                    if _is_model_not_found(probe_exc):
                        errors.append(f"{name}: {probe_exc}")
                        continue
                    raise

            kwargs = {"model": name, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            st.session_state["gemini_model"] = name
            return response
        except Exception as exc:
            if _is_model_not_found(exc) or "model" in str(exc).lower():
                errors.append(f"{name}: {exc}")
                continue
            raise

    detail = " | ".join(errors) if errors else "no generateContent models were returned"
    raise RuntimeError(
        f"No working Gemini Flash/Pro model (404/NOT_FOUND fallbacks exhausted): {detail}"
    )


@st.cache_data
def load_food_database() -> pd.DataFrame:
    """Load the local Indian food CSV once per session cache."""
    df = pd.read_csv(FOODS_CSV)
    df["FODMAP_Level"] = df["FODMAP_Level"].astype(str).str.strip()
    df["Glycemic_Index"] = pd.to_numeric(df["Glycemic_Index"], errors="coerce")
    df["Sat_Fat_g"] = pd.to_numeric(df["Sat_Fat_g"], errors="coerce")
    df["B12_mcg"] = pd.to_numeric(df["B12_mcg"], errors="coerce")
    df["VitD_IU"] = pd.to_numeric(df["VitD_IU"], errors="coerce")
    return df.dropna(subset=["Food_Name"])


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Lightweight local text extract (pypdf) — used as a fallback if inline PDF fails."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def parse_lab_report_with_gemini(client: genai.Client, pdf_bytes: bytes) -> dict:
    """
    Ask Gemini to read the lab PDF and return structured biomarker flags.
    Falls back to sending pypdf-extracted text if the binary PDF call fails.
    """
    prompt = """
You are a clinical lab-report parser. Read the attached blood test / lab report.
Extract HbA1c, Vitamin B12, and Vitamin D if present.

Return ONLY valid JSON with this schema:
{
  "hba1c_percent": number or null,
  "hba1c_status": "normal" | "abnormal" | "not_found",
  "b12_value": number or null,
  "b12_unit": string or null,
  "b12_status": "normal" | "abnormal" | "not_found",
  "vitd_value": number or null,
  "vitd_unit": string or null,
  "vitd_status": "normal" | "abnormal" | "not_found",
  "notes": "one short sentence for the clinician-facing banner"
}

Status rules (use report reference ranges if printed; otherwise use):
- HbA1c abnormal if >= 5.7%
- Vitamin B12 abnormal if below the report's lower limit, or < 200 pg/mL (or < 148 pmol/L)
- Vitamin D abnormal if below the report's lower limit, or < 20 ng/mL (or < 50 nmol/L)
Do not diagnose disease. Do not invent values that are not in the report.
"""

    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                prompt,
            ],
            config=config,
        )
        return json.loads(response.text or "{}")
    except Exception:
        # Fallback: send extracted text instead of raw PDF bytes
        text = extract_pdf_text(pdf_bytes)
        if not text:
            raise RuntimeError("Could not read PDF as a document or as text.")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt
                + "\n\n--- Extracted lab report text ---\n"
                + text[:20000]
            ],
            config=config,
        )
        return json.loads(response.text or "{}")


def apply_clinical_filters(
    df: pd.DataFrame,
    ibs: bool,
    prediabetes: bool,
    nafld: bool,
    micronutrient_priority: bool,
) -> pd.DataFrame:
    """
    Deterministic hard filters (symbolic layer). Gemini never sees rejected foods.
    Vitamin deficiency does not drop rows; it re-ranks nutrient-dense items first.
    """
    safe = df.copy()

    if ibs:
        safe = safe[safe["FODMAP_Level"].str.lower() == "low"]

    if prediabetes:
        safe = safe[safe["Glycemic_Index"] <= 55]

    if nafld:
        safe = safe[safe["Sat_Fat_g"] <= 3.0]

    if micronutrient_priority:
        # Composite density score: B12 (mcg) + Vit D scaled down from IU
        safe = safe.assign(
            _micronutrient_score=safe["B12_mcg"].fillna(0)
            + (safe["VitD_IU"].fillna(0) / 40.0)
        )
        safe = safe.sort_values("_micronutrient_score", ascending=False)
        safe = safe.drop(columns=["_micronutrient_score"])
    else:
        safe = safe.sort_values(["Category", "Food_Name"])

    return safe.reset_index(drop=True)


def generate_meal_plan(client: genai.Client, approved_foods: list[str], conditions: list[str]) -> str:
    """Zero-hallucination meal plan: Gemini may use ONLY the Pandas-approved list."""
    food_list = ", ".join(approved_foods)
    condition_line = ", ".join(conditions) if conditions else "general healthy Indian household diet"

    prompt = f"""You are a clinical Indian dietitian. Using ONLY the following pre-approved safe ingredients: {food_list}, construct a 1-day Indian meal plan (Breakfast, Lunch, Dinner). Do NOT include any ingredient outside this list under any circumstance. Provide simple cooking instructions and explain why this plan is safe.

Active clinical considerations from the constraint engine: {condition_line}.
Respond in Markdown with Breakfast, Lunch, Dinner, and a short safety explanation.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )
    return (response.text or "").strip()


def render_biomarker_banner(lab: dict) -> None:
    """Show a compact clinical summary of detected lab values."""
    if not lab:
        return

    st.subheader("Detected biomarkers")
    notes = lab.get("notes") or "Lab values extracted from the uploaded report."
    st.info(notes)

    c1, c2, c3 = st.columns(3)

    def _metric(col, title: str, value, unit: str | None, status: str | None) -> None:
        display = "Not found"
        if value is not None and value != "":
            unit_s = f" {unit}" if unit else ""
            display = f"{value}{unit_s}"
        status = (status or "not_found").lower()
        delta = None
        if status == "abnormal":
            delta = "Abnormal"
        elif status == "normal":
            delta = "Within typical range"
        with col:
            st.metric(title, display, delta=delta, delta_color="inverse" if status == "abnormal" else "normal")

    _metric(
        c1,
        "HbA1c",
        lab.get("hba1c_percent"),
        "%",
        lab.get("hba1c_status"),
    )
    _metric(
        c2,
        "Vitamin B12",
        lab.get("b12_value"),
        lab.get("b12_unit"),
        lab.get("b12_status"),
    )
    _metric(
        c3,
        "Vitamin D",
        lab.get("vitd_value"),
        lab.get("vitd_unit"),
        lab.get("vitd_status"),
    )

    hints = []
    if str(lab.get("hba1c_status", "")).lower() == "abnormal":
        hints.append("Consider enabling **Pre-Diabetes / Insulin Resistance** (Low GI ≤ 55).")
    if str(lab.get("b12_status", "")).lower() == "abnormal" or str(lab.get("vitd_status", "")).lower() == "abnormal":
        hints.append("Consider enabling **Vitamin B12 / D Deficiency** to rank nutrient-dense foods first.")
    if hints:
        st.caption("Suggested filters (manual): " + " ".join(hints))


# ---------------------------------------------------------------------------
# Sidebar — patient clinical profile
# ---------------------------------------------------------------------------
client = get_gemini_client()

with st.sidebar:
    st.title("Patient Clinical Profile")
    st.caption("Hard filters run in Pandas. Gemini only sees foods that survive.")

    uploaded_pdf = st.file_uploader(
        "Upload Blood Test PDF (Lab Report)",
        type=["pdf"],
        help="Gemini extracts HbA1c, B12, and Vitamin D when possible.",
    )

    st.markdown("**Manual condition toggles**")
    ibs = st.checkbox("IBS / Gut Inflammation (Triggers Low FODMAP filter)")
    prediabetes = st.checkbox("Pre-Diabetes / Insulin Resistance (Triggers Low GI <= 55 filter)")
    nafld = st.checkbox("Fatty Liver (NAFLD) Precautions (Triggers Saturated Fat <= 3.0g filter)")
    micronutrient = st.checkbox("Vitamin B12 / D Deficiency (Prioritizes nutrient-dense foods)")

    if client is None:
        st.warning("Set `GEMINI_API_KEY` in the `.env` file to enable PDF parsing and meal plans.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("NutriSolver")
st.markdown("**Clinical Diet Optimization Co-Pilot** — Indian household diets, neuro-symbolic safety layer.")

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    cache_key = f"{uploaded_pdf.name}-{len(pdf_bytes)}"

    if st.session_state.get("_lab_pdf_key") != cache_key:
        st.session_state["_lab_pdf_key"] = cache_key
        st.session_state["lab_result"] = None
        if client is None:
            st.session_state["lab_error"] = "Missing Gemini API key."
        else:
            with st.spinner("Parsing lab report with Gemini…"):
                try:
                    st.session_state["lab_result"] = parse_lab_report_with_gemini(client, pdf_bytes)
                    st.session_state["lab_error"] = None
                except Exception as exc:
                    st.session_state["lab_result"] = None
                    st.session_state["lab_error"] = str(exc)

    if st.session_state.get("lab_error"):
        st.error(f"Could not parse the lab PDF: {st.session_state['lab_error']}")
    else:
        render_biomarker_banner(st.session_state.get("lab_result") or {})

# Constraint engine
try:
    foods = load_food_database()
except FileNotFoundError:
    st.error(f"Food database not found at `{FOODS_CSV}`.")
    st.stop()

safe_foods = apply_clinical_filters(foods, ibs, prediabetes, nafld, micronutrient)

st.subheader("Mathematically Approved Safe Foods")
active = []
if ibs:
    active.append("Low FODMAP only")
if prediabetes:
    active.append("GI ≤ 55")
if nafld:
    active.append("Sat fat ≤ 3.0 g")
if micronutrient:
    active.append("Ranked by B12 + Vitamin D density")
st.caption("Active symbolic constraints: " + (", ".join(active) if active else "none — showing full local database."))

if safe_foods.empty:
    st.error("No foods remain after the selected hard filters. Relax one condition or expand the CSV.")
else:
    st.dataframe(safe_foods, use_container_width=True, hide_index=True)
    st.caption(f"{len(safe_foods)} of {len(foods)} foods passed the Pandas constraint engine.")

st.divider()
st.subheader("Clinical meal plan")
generate = st.button("Generate Clinical Meal Plan", type="primary", disabled=safe_foods.empty)

if generate:
    if client is None:
        st.error("Add a valid `GEMINI_API_KEY` to `.env`, then restart the app.")
    elif safe_foods.empty:
        st.error("Cannot generate a plan with an empty approved-food list.")
    else:
        approved_names = safe_foods["Food_Name"].astype(str).tolist()
        conditions = []
        if ibs:
            conditions.append("IBS / low-FODMAP")
        if prediabetes:
            conditions.append("pre-diabetes / insulin resistance")
        if nafld:
            conditions.append("NAFLD / saturated-fat limit")
        if micronutrient:
            conditions.append("B12 and/or Vitamin D deficiency — prefer nutrient-dense listed foods")

        with st.spinner("Generating a constrained Indian meal plan…"):
            try:
                st.session_state["meal_plan"] = generate_meal_plan(client, approved_names, conditions)
                st.session_state["meal_plan_error"] = None
            except Exception as exc:
                st.session_state["meal_plan"] = None
                st.session_state["meal_plan_error"] = str(exc)

if st.session_state.get("meal_plan_error"):
    st.error(f"Meal plan generation failed: {st.session_state['meal_plan_error']}")

if st.session_state.get("meal_plan"):
    st.markdown(st.session_state["meal_plan"])

st.markdown("---")
st.caption(
    "Educational co-pilot only. Lab flags and meal plans are not a medical diagnosis or a substitute for a registered dietitian or physician."
)
