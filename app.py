import io
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import openpyxl
import streamlit as st
from pydantic import BaseModel
from google import genai
from google.genai import types

st.set_page_config(page_title="SKK Migas Auto-Cataloging System", layout="wide")

MS_FILE = "Material_Standard_26112014119179.xlsx"

# --- 0. Pembaca Secrets yang Aman (Tidak Error Jika Secrets Kosong) ---
def get_secret_safe(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

# --- 1. Pembaca Master Excel Tunggal SKK Migas ---
def read_skk_xlsx(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(filepath, dtype=str)
        if not df.empty:
            return df
    except Exception:
        pass

    with zipfile.ZipFile(filepath, "r") as z:
        xml_content = z.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(xml_content)
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows = []
    for row in root.findall(".//main:sheetData/main:row", ns):
        r_vals = []
        for c in row.findall("main:c", ns):
            t_el = c.find(".//main:t", ns)
            v_el = c.find("main:v", ns)
            if t_el is not None:
                r_vals.append(t_el.text)
            elif v_el is not None:
                r_vals.append(v_el.text)
            else:
                r_vals.append(None)
        rows.append(r_vals)
    return pd.DataFrame(rows[1:], columns=rows[0])

@st.cache_data
def load_master_data():
    df_ms = read_skk_xlsx(MS_FILE).fillna("")
    df_ms.columns = [c.strip().upper() for c in df_ms.columns]
    df_ms["NOUN"] = df_ms["NOUN"].astype(str).str.strip().str.upper()
    df_ms["MODIFIER"] = df_ms["MODIFIER"].astype(str).str.strip().str.upper()
    df_ms["SEQ"] = df_ms["SEQ"].astype(str).str.strip().astype(int)
    df_ms["MANDATORY"] = df_ms["MANDATORY"].astype(str).str.strip().str.upper()
    df_ms["ATTRIBUTE"] = df_ms["ATTRIBUTE"].astype(str).str.strip().str.upper()

    df_nm = (
        df_ms[["NOUN", "MODIFIER"]]
        .drop_duplicates()
        .sort_values(["NOUN", "MODIFIER"])
        .reset_index(drop=True)
    )
    return df_nm, df_ms

# --- 2. Pencocokan Langsung ke Tabel Master (Lapisan 1 - Tanpa API Key) ---
ALIAS_RULES = [
    (r"\bCROSS\s*OVER\b", ("CROSS-OVER", "--", "Cocok aturan master CROSS-OVER")),
    (r"\bCENTRALI[SZ]ER\b.*\bTUBING\b", ("CENTRALIZER", "TUBING", "Cocok master CENTRALIZER | TUBING")),
    (r"\bCENTRALI[SZ]ER\b", ("CENTRALIZER", "CASING", "Cocok master CENTRALIZER | CASING")),
    (r"\bRTD\b", ("DETECTOR", "RESISTANCE TEMPERATURE", "Singkatan RTD -> DETECTOR | RESISTANCE TEMPERATURE")),
    (r"\bGASKET\b.*\b(SPW|SPIRAL\s*WOUND)\b", ("GASKET", "SPIRAL WOUND", "Cocok master GASKET | SPIRAL WOUND")),
    (r"\bGASKET\b.*\bRING\s*JOINT\b", ("GASKET", "RING JOINT", "Cocok master GASKET | RING JOINT")),
    (r"\bGAUGE\b.*\bPRESS", ("GAUGE", "PRESSURE", "Cocok master GAUGE | PRESSURE")),
    (r"\bGAUGE\b.*\bTEMP", ("GAUGE", "TEMPERATURE", "Cocok master GAUGE | TEMPERATURE")),
    (r"\bSAFETY\s*GOGGLES\b", ("GOGGLES", "SAFETY", "Cocok master GOGGLES | SAFETY")),
    (r"\bCIRCUIT\s*BREAKER\b", ("BREAKER", "CIRCUIT", "Cocok master BREAKER | CIRCUIT")),
    (r"\bLANDING\s*COLLARS?\b", ("COLLAR", "LANDING", "Cocok master COLLAR | LANDING")),
    (r"\bCASING\b", ("CASING", "OCTG", "Cocok master CASING | OCTG")),
    (r"\bTUBING\b", ("TUBING", "OCTG", "Cocok master TUBING | OCTG")),
]

def match_from_master_direct(short_desc: str, long_desc: str, df_nm: pd.DataFrame):
    combined = f"{short_desc} , {long_desc}".upper()
    valid_pairs = set(zip(df_nm["NOUN"], df_nm["MODIFIER"]))

    for pattern, (n_target, m_target, note) in ALIAS_RULES:
        if re.search(pattern, combined) and (n_target, m_target) in valid_pairs:
            return n_target, m_target, 98, note

    parts = [p.strip() for p in short_desc.upper().split(",") if p.strip()]
    if not parts:
        return None

    first_token = parts[0]
    noun_matches = df_nm[df_nm["NOUN"] == first_token]
    if not noun_matches.empty:
        mods = noun_matches["MODIFIER"].tolist()
        for mod in sorted(mods, key=len, reverse=True):
            if mod != "--" and re.search(rf"\b{re.escape(mod)}\b", combined):
                return first_token, mod, 99, f"Cocok langsung master ({first_token} | {mod})"
        if len(mods) == 1:
            return first_token, mods[0], 95, f"Satu-satunya pasangan master untuk {first_token}"

    return None

# --- 3. Skema Output Terstruktur AI ---
class CandidateNM(BaseModel):
    noun: str
    modifier: str
    confidence: int
    reason: str

class ClassificationResult(BaseModel):
    top_candidates: list[CandidateNM]
    needs_review: bool

class BatchItemDecision(BaseModel):
    row_index: int
    noun: str
    modifier: str
    confidence: int
    reason: str

class BatchClassificationResult(BaseModel):
    items: list[BatchItemDecision]

class ExtractedAttr(BaseModel):
    seq: int
    attribute: str
    value: str

class ExtractionResult(BaseModel):
    attributes: list[ExtractedAttr]
    short_description: str
    long_description: str

# --- 4. Fungsi Panggilan Gemini API ---
def get_gemini_client():
    api_key = st.session_state.get("ui_api_key", "").strip() or get_secret_safe("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Silakan masukkan Gemini API Key di kotak pengaturan sebelah kiri (Sidebar) terlebih dahulu.")
        st.stop()
    return genai.Client(api_key=api_key)

def get_model_name():
    return st.session_state.get("ui_model_name", "").strip() or get_secret_safe("GEMINI_MODEL", "gemini-2.5-flash")

def filter_relevant_nm(texts: list[str], df_nm: pd.DataFrame) -> list[str]:
    nm_pairs = list(zip(df_nm["NOUN"], df_nm["MODIFIER"]))
    tokens = set()
    for t in texts:
        cleaned = t.upper().replace(",", " ").replace(";", " ").replace("/", " ").replace("-", " ")
        for w in cleaned.split():
            w = w.strip("().:\"")
            if len(w) >= 3:
                tokens.add(w)
    filtered = [f"{n} | {m}" for n, m in nm_pairs if any(tok in n or tok in m for tok in tokens)]
    if "SPARE PARTS | --" not in filtered:
        filtered.append("SPARE PARTS | --")
    return filtered if len(filtered) >= 15 else [f"{n} | {m}" for n, m in nm_pairs]

def classify_material(raw_text: str, df_nm: pd.DataFrame) -> ClassificationResult:
    client = get_gemini_client()
    filtered = filter_relevant_nm([raw_text], df_nm)
    prompt = (
        "Anda adalah cataloguer senior SKK Migas.\n"
        "Tentukan 3 kandidat (NOUN | MODIFIER) paling tepat dan SPESIFIK dari DAFTAR RESMI.\n"
        "Utamakan NOUN & MODIFIER spesifik sesuai benda utama; hindari 'SPARE PARTS | --' jika benda punya Noun standar.\n\n"
        f"DESKRIPSI:\n\"{raw_text}\"\n\n"
        f"DAFTAR RESMI:\n{chr(10).join(filtered)}"
    )
    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificationResult,
            temperature=0.05,
        ),
    )
    return ClassificationResult.model_validate_json(response.text)

def classify_material_batch(batch_rows: list[dict], df_nm: pd.DataFrame) -> dict[int, BatchItemDecision]:
    client = get_gemini_client()
    texts = [f"{r['short']} {r['long']}" for r in batch_rows]
    filtered = filter_relevant_nm(texts, df_nm)
    items_block = "\n".join(
        f"- row_index: {r['row_index']} | Short: \"{r['short']}\" | Long: \"{r['long']}\""
        for r in batch_rows
    )
    prompt = (
        "Anda adalah cataloguer senior SKK Migas.\n"
        "Klasifikasikan setiap item ke pasangan (NOUN | MODIFIER) paling spesifik dari DAFTAR RESMI.\n"
        "Utamakan NOUN & MODIFIER spesifik sesuai benda utama; hindari 'SPARE PARTS | --' jika benda punya Noun standar.\n\n"
        f"DAFTAR INPUT:\n{items_block}\n\n"
        f"DAFTAR RESMI:\n{chr(10).join(filtered)}"
    )
    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchClassificationResult,
            temperature=0.05,
        ),
    )
    parsed = BatchClassificationResult.model_validate_json(response.text)
    return {item.row_index: item for item in parsed.items}

def extract_attributes(raw_text: str, noun: str, modifier: str, schema_df: pd.DataFrame) -> ExtractionResult:
    client = get_gemini_client()
    schema_lines = [
        f"SEQ {r['SEQ']} | Mandatory: {r['MANDATORY']} | Attribute: {r['ATTRIBUTE']}"
        for _, r in schema_df.iterrows()
    ]
    prompt = (
        "Ekstrak nilai spesifikasi dari deskripsi mentah ke slot atribut baku SKK Migas (UPPERCASE).\n"
        "Kosongkan ('') jika tidak tertulis. Susun short_description (max 40 char) dan long_description baku.\n\n"
        f"NOUN = {noun}, MODIFIER = {modifier}\n"
        f"DESKRIPSI: \"{raw_text}\"\n\n"
        f"SKEMA ATRIBUT:\n{chr(10).join(schema_lines)}"
    )
    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractionResult,
            temperature=0.1,
        ),
    )
    return ExtractionResult.model_validate_json(response.text)

# --- 5. Generator File 1 & File 2 ---
def build_file1_buffer(df_rows: pd.DataFrame) -> io.BytesIO:
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Source_Data_Upload"
    ws1.append([
        "No.", "PSC Material Code", "Noun", "Modifier",
        "Material - Short Description", "Material - Long Description", "Uom"
    ])
    for idx, row in df_rows.reset_index(drop=True).iterrows():
        mat_code = str(row.get("PSC Material Code", "")).strip().replace("\t", "")[:20]
        noun = str(row.get("Noun", "")).strip().upper()[:30]
        modifier = str(row.get("Modifier", "")).strip().upper()[:30]
        short_desc = str(row.get("Material - Short Description", "")).strip().replace("\t", "")[:100]
        long_desc = str(row.get("Material - Long Description", "")).strip().replace("\t", "")[:3000]
        uom = str(row.get("Uom", "")).strip().upper()[:10]
        ws1.append([idx + 1, mat_code, noun, modifier, short_desc, long_desc, uom])

    ws2 = wb.create_sheet(title="Source_Data_Upload_Format")
    format_rows = [
        ["INTEGER", "ALPHANUMERIC", "ALPHANUMERIC", "ALPHANUMERIC", "ALPHANUMERIC", "ALPHANUMERIC", "ALPHANUMERIC", "HEADER DATA", "DATA TYPE"],
        ["No. of the material data ", "The Code/KEYMAP# of the material", "NOUN as per the CLASS master", "MODIFIER as per the CLASS master", "Short description of the material", "Long description of the material", "Uom of the material", None, "DESCRIPTION"],
        ["2", "20", "30", "30", "100", "3000", "10", None, "SIZE"],
        ["No.", "PSC Material code", "Noun", "Modifier", "Material - Short Description", "Material - Long Description", "Uom", None, "HEADER"],
        [1, "100001", "SPARE PARTS", "--", "PLATE A 5mm", "CONTACTOR:AUX,50/60HZ,3NO 2NC; MANUFACTURER: KONECRANES; P/N: CAD32V7", "KGS", None, "EXAMPLE 1"],
        [2, "100003", "SPARE PARTS", "--", "PLATE A 6mm", "BRAKE; MANUFACTURER: KONECRANES; P/N: NM38720NR2V", "KGS", None, "EXAMPLE 2"],
        [3, "100004", "SPARE PARTS", "--", "PLATE A 7mm", "BRAKE; MANUFACTURER: KONECRANES; P/N: NM38730NR2V", "KGS", None, "EXAMPLE 3"],
        [4, "100005", "SPARE PARTS", "--", "PLATE A 8mm", "BRAKE:; MANUFACTURER: KONECRANES; P/N: NM38740NR2V", "KGS", None, "EXAMPLE 4"],
        [5, "100006", "SPARE PARTS", "--", "PLATE A 9mm", "SET:BRAKE DISC; MANUFACTURER: KONECRANES; P/N: SP-NM38730JPMV", "KGS", None, "EXAMPLE 5"],
    ]
    for r in format_rows:
        ws2.append(r)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

def build_file2_buffer(items: list[dict], df_ms: pd.DataFrame) -> io.BytesIO:
    rows_file2 = []
    seq_counter_f2 = 1
    for item in items:
        psc_code = str(item["psc_code"]).strip()
        mat_code = str(item.get("psc_material_code", "")).strip()
        ref_no = str(item["ref_no"]).strip().replace("\t", "")
        noun = str(item["noun"]).strip().upper()
        modifier = str(item["modifier"]).strip().upper()
        uom = str(item["uom"]).strip().upper()
        long_desc = str(item["long_description"]).strip().upper()[:3000]

        schema_df = df_ms[(df_ms["NOUN"] == noun) & (df_ms["MODIFIER"] == modifier)].sort_values("SEQ")
        val_map = {int(a["seq"]): str(a["value"]).strip().upper().replace("\t", "") for a in item["attributes"]}

        for _, s_row in schema_df.iterrows():
            seq = int(s_row["SEQ"])
            rows_file2.append({
                "No.": seq_counter_f2,
                "PSC Code": psc_code,
                "PSC Material Code": mat_code,
                "Ref.No": ref_no,
                "Noun": noun,
                "Modifier": modifier,
                "Sequence Number": seq,
                "Attribute": s_row["ATTRIBUTE"],
                "Value": val_map.get(seq, ""),
                "Remarks": "",
                "Mandatory": s_row["MANDATORY"],
                "Uom": uom if seq == 1 else "",
                "Material - Long Description": long_desc if seq == 1 else "",
            })
            seq_counter_f2 += 1

    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
        pd.DataFrame(rows_file2).to_excel(writer, sheet_name="Data_Extraction_Upload", index=False)
    buf2.seek(0)
    return buf2

# --- 6. Antarmuka Utama ---
df_nm, df_ms = load_master_data()

if "single_items" not in st.session_state:
    st.session_state.single_items = []
if "file1_records" not in st.session_state:
    st.session_state.file1_records = []
if "file2_items" not in st.session_state:
    st.session_state.file2_items = []

with st.sidebar:
    st.title("Navigasi Menu")
    menu_pilihan = st.radio(
        "Pilih Mode Kerja:",
        [
            "Menu 1: Analisa & Ekstrak Satuan",
            "Menu 2: Batch Tahap 1 (Source_Data_Upload)",
            "Menu 3: Batch Tahap 2 (Data_Extraction_Upload)",
        ],
    )
    st.divider()
    st.header("Konfigurasi API & PSC")
    st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Tempel AIzaSy... di sini",
        key="ui_api_key",
    )
    st.selectbox(
    "Model Gemini",
    ["gemini-3.8-flash", "gemini-3.8-pro", "gemini-3.0-flash"],
    key="ui_model_name",
)
    default_psc = st.text_input("PSC Code", value="A095")
    st.success(f"Master Tunggal Aktif:\n- {len(df_nm):,} Noun-Modifier\n- {len(df_ms):,} Baris Atribut")

if menu_pilihan == "Menu 1: Analisa & Ekstrak Satuan":
    st.title("Menu 1: Analisa & Ekstrak Satuan")
    t1_in, t1_rev = st.tabs(["1. Analisa & Klasifikasi Material", "2. Review & Download Hasil Satuan"])

    with t1_in:
        c1, c2, c3 = st.columns([2, 2, 1])
        mat_code = c1.text_input("PSC Material Code (Wajib)", placeholder="Contoh: 51068322")
        ref_no = c2.text_input("Ref.No", placeholder="Contoh: A095_1367818")
        uom = c3.text_input("UoM", value="EA")
        raw_desc = st.text_area("Deskripsi Material Mentah:", height=120)

        if st.button("Cari Rekomendasi Noun - Modifier", type="primary") and raw_desc.strip():
            with st.spinner("Mencocokkan dengan master & AI..."):
                st.session_state.last_class_m1 = classify_material(raw_desc, df_nm)
                st.session_state.last_input_m1 = {
                    "psc_code": default_psc,
                    "psc_material_code": mat_code,
                    "ref_no": ref_no,
                    "uom": uom,
                    "raw_desc": raw_desc,
                }

        if "last_class_m1" in st.session_state:
            res = st.session_state.last_class_m1
            options = [f"{c.noun} | {c.modifier}  —  ({c.confidence}% | {c.reason})" for c in res.top_candidates]
            options.append("Pilih Manual...")
            choice = st.radio("Kandidat Klasifikasi:", options, key="rad_m1")

            if choice.startswith("Pilih Manual"):
                all_nm = [f"{r['NOUN']} | {r['MODIFIER']}" for _, r in df_nm.iterrows()]
                manual_pick = st.selectbox("Cari Noun | Modifier:", all_nm, key="sel_m1")
                sel_noun, sel_mod = [x.strip() for x in manual_pick.split("|")]
            else:
                sel_noun, sel_mod = [x.strip() for x in choice.split("—")[0].split("|")]

            if st.button("Ekstrak Atribut & Simpan ke Daftar"):
                schema_df = df_ms[(df_ms["NOUN"] == sel_noun) & (df_ms["MODIFIER"] == sel_mod)].sort_values("SEQ")
                inp = st.session_state.last_input_m1
                ext = extract_attributes(inp["raw_desc"], sel_noun, sel_mod, schema_df)
                attr_map = {a.seq: a.value for a in ext.attributes}
                full_attrs = [
                    {
                        "seq": int(r["SEQ"]),
                        "mandatory": r["MANDATORY"],
                        "attribute": r["ATTRIBUTE"],
                        "value": attr_map.get(int(r["SEQ"]), ""),
                    }
                    for _, r in schema_df.iterrows()
                ]
                st.session_state.single_items.append({
                    "psc_code": inp["psc_code"],
                    "psc_material_code": inp["psc_material_code"],
                    "ref_no": inp["ref_no"],
                    "noun": sel_noun,
                    "modifier": sel_mod,
                    "uom": inp["uom"],
                    "raw_desc": inp["raw_desc"],
                    "short_description": ext.short_description,
                    "long_description": ext.long_description,
                    "attributes": full_attrs,
                })
                st.success("Tersimpan! Buka tab '2. Review & Download Hasil Satuan'.")

    with t1_rev:
        if not st.session_state.single_items:
            st.info("Belum ada material.")
        else:
            for i, item in enumerate(st.session_state.single_items):
                df_attr = pd.DataFrame(item["attributes"])
                missing_m = df_attr[(df_attr["mandatory"] == "M") & (df_attr["value"].astype(str).str.strip() == "")]
                lbl = f"⚠️ {len(missing_m)} Mandatory Kosong" if len(missing_m) > 0 else "✅ Lengkap"
                with st.expander(f"{i+1}. [{item['psc_material_code']}] {item['noun']} - {item['modifier']} | {lbl}", expanded=True):
                    c_a, c_b, c_c = st.columns(3)
                    item["psc_material_code"] = c_a.text_input("PSC Material Code", item["psc_material_code"], key=f"m1_mc_{i}")
                    item["ref_no"] = c_b.text_input("Ref.No", item["ref_no"], key=f"m1_rn_{i}")
                    item["uom"] = c_c.text_input("UoM", item["uom"], key=f"m1_um_{i}")
                    item["short_description"] = st.text_input("Short Description", item["short_description"], key=f"m1_sd_{i}")
                    item["long_description"] = st.text_area("Long Description", item["long_description"], key=f"m1_ld_{i}")
                    edited_df = st.data_editor(df_attr, use_container_width=True, disabled=["seq", "mandatory", "attribute"], key=f"m1_ed_{i}")
                    item["attributes"] = edited_df.to_dict(orient="records")

            df_s_f1 = pd.DataFrame([
                {
                    "PSC Material Code": it["psc_material_code"],
                    "Noun": it["noun"],
                    "Modifier": it["modifier"],
                    "Material - Short Description": it["short_description"],
                    "Material - Long Description": it["long_description"],
                    "Uom": it["uom"],
                }
                for it in st.session_state.single_items
            ])
            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button("Download File 1 (Source_Data_Upload.xlsx)", build_file1_buffer(df_s_f1), "Source_Data_Upload.xlsx", type="primary")
            col_dl2.download_button("Download File 2 (Data_Extraction_Upload.xlsx)", build_file2_buffer(st.session_state.single_items, df_ms), "Data_Extraction_Upload.xlsx", type="primary")

elif menu_pilihan == "Menu 2: Batch Tahap 1 (Source_Data_Upload)":
    st.title("Menu 2: Batch Upload Tahap 1 (Source_Data_Upload)")
    uploaded_raw = st.file_uploader("Upload File Excel Material Mentah (.xlsx)", type=["xlsx"], key="up_raw")
    if uploaded_raw is not None:
        df_in = pd.read_excel(uploaded_raw, dtype=str).fillna("")
        st.write(f"Terbaca **{len(df_in)} baris** material:")
        st.dataframe(df_in.head(10), use_container_width=True)

        cols = df_in.columns.tolist()
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        col_code = c_m1.selectbox("Kolom PSC Material Code (Wajib)", cols, index=0)
        col_short = c_m2.selectbox("Kolom Short Description", cols, index=min(1, len(cols)-1))
        col_long = c_m3.selectbox("Kolom Long Description", cols, index=min(2, len(cols)-1))
        col_uom = c_m4.selectbox("Kolom Uom", cols, index=min(3, len(cols)-1))

        if st.button("Proses Klasifikasi Massal Sekarang", type="primary"):
            total = len(df_in)
            results = [None] * total
            unmatched_rows = []
            valid_pairs = set(zip(df_nm["NOUN"], df_nm["MODIFIER"]))

            # Lapisan 1: Pencocokan langsung dengan master SKK Migas (tidak butuh API key)
            for idx in range(total):
                r = df_in.iloc[idx]
                m_code = str(r[col_code]).strip()
                s_desc = str(r[col_short]).strip()
                l_desc = str(r[col_long]).strip()
                uom_val = str(r[col_uom]).strip().upper() or "EA"

                direct = match_from_master_direct(s_desc, l_desc, df_nm)
                if direct:
                    noun_val, mod_val, conf, note = direct
                    results[idx] = {
                        "PSC Material Code": m_code,
                        "Noun": noun_val,
                        "Modifier": mod_val,
                        "Material - Short Description": s_desc,
                        "Material - Long Description": l_desc,
                        "Uom": uom_val,
                        "Keyakinan AI (%)": conf,
                        "Catatan AI": note,
                    }
                else:
                    unmatched_rows.append({
                        "row_index": idx,
                        "short": s_desc,
                        "long": l_desc,
                    })

            # Lapisan 2: Sisa item yang butuh analisis AI diproses per batch 10
            if unmatched_rows:
                has_key = bool(st.session_state.get("ui_api_key", "").strip() or get_secret_safe("GEMINI_API_KEY", ""))
                if not has_key:
                    st.warning(f"{total - len(unmatched_rows)} item berhasil dicocokkan langsung ke master. Masukkan Gemini API Key di Sidebar untuk memproses {len(unmatched_rows)} item sisanya dengan AI.")
                    for item_in in unmatched_rows:
                        idx = item_in["row_index"]
                        r = df_in.iloc[idx]
                        results[idx] = {
                            "PSC Material Code": str(r[col_code]).strip(),
                            "Noun": "SPARE PARTS",
                            "Modifier": "--",
                            "Material - Short Description": str(r[col_short]).strip(),
                            "Material - Long Description": str(r[col_long]).strip(),
                            "Uom": str(r[col_uom]).strip().upper() or "EA",
                            "Keyakinan AI (%)": 0,
                            "Catatan AI": "Isi Gemini API Key di Sidebar untuk klasifikasi AI",
                        }
                else:
                    prog = st.progress(0)
                    status_txt = st.empty()
                    batch_size = 10
                    tot_ai = len(unmatched_rows)

                    for b_start in range(0, tot_ai, batch_size):
                        b_end = min(b_start + batch_size, tot_ai)
                        status_txt.text(f"AI menganalisis {b_start+1} s/d {b_end} dari {tot_ai} item tersisa...")
                        chunk = unmatched_rows[b_start:b_end]

                        try:
                            decisions = classify_material_batch(chunk, df_nm)
                        except Exception as e:
                            st.error(f"Gagal memanggil Gemini API ({get_model_name()}): {e}")
                            decisions = {}

                        for item_in in chunk:
                            idx = item_in["row_index"]
                            r = df_in.iloc[idx]
                            m_code = str(r[col_code]).strip()
                            s_desc = str(r[col_short]).strip()
                            l_desc = str(r[col_long]).strip()
                            uom_val = str(r[col_uom]).strip().upper() or "EA"

                            dec = decisions.get(idx)
                            if dec and (dec.noun.strip().upper(), dec.modifier.strip().upper()) in valid_pairs:
                                noun_val = dec.noun.strip().upper()
                                mod_val = dec.modifier.strip().upper()
                                conf = dec.confidence
                                note = dec.reason
                            elif dec:
                                matched_mods = df_nm[df_nm["NOUN"] == dec.noun.strip().upper()]["MODIFIER"].tolist()
                                if matched_mods:
                                    noun_val = dec.noun.strip().upper()
                                    mod_val = matched_mods[0]
                                    conf = dec.confidence
                                    note = f"{dec.reason} (Modifier disesuaikan)"
                                else:
                                    noun_val, mod_val, conf, note = "SPARE PARTS", "--", 50, "Perlu review manual"
                            else:
                                noun_val, mod_val, conf, note = "SPARE PARTS", "--", 0, "Cek API Key"

                            results[idx] = {
                                "PSC Material Code": m_code,
                                "Noun": noun_val,
                                "Modifier": mod_val,
                                "Material - Short Description": s_desc,
                                "Material - Long Description": l_desc,
                                "Uom": uom_val,
                                "Keyakinan AI (%)": conf,
                                "Catatan AI": note,
                            }
                        prog.progress(b_end / tot_ai)
                        time.sleep(0.2)
                    status_txt.success("Selesai memproses seluruh material!")

            st.session_state.file1_records = results

    if st.session_state.file1_records:
        df_res_preview = pd.DataFrame(st.session_state.file1_records)
        sp_count = (df_res_preview["Noun"] == "SPARE PARTS").sum()
        mod_filled = (df_res_preview["Modifier"] != "--").sum()
        st.info(f"Ringkasan: **{mod_filled} item** terisi Modifier spesifik, **{sp_count} item** masuk SPARE PARTS | --.")

        df_f1_edit = st.data_editor(df_res_preview, use_container_width=True, num_rows="dynamic", key="editor_f1_batch")
        buf_f1 = build_file1_buffer(df_f1_edit)
        st.download_button(
            label="Download File 1 (Source_Data_Upload.xlsx)",
            data=buf_f1,
            file_name="Source_Data_Upload.xlsx",
            type="primary",
        )

elif menu_pilihan == "Menu 3: Batch Tahap 2 (Data_Extraction_Upload)":
    st.title("Menu 3: Batch Ekstraksi Tahap 2 (Data_Extraction_Upload)")
    uploaded_f2_src = st.file_uploader("Upload Excel Daftar Material Tahap 2 (.xlsx)", type=["xlsx"], key="up_f2")
    if uploaded_f2_src is not None:
        df_in2 = pd.read_excel(uploaded_f2_src, dtype=str).fillna("")
        st.dataframe(df_in2.head(5), use_container_width=True)
        cols2 = df_in2.columns.tolist()
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        c2_code = k1.selectbox("PSC Material Code", cols2, index=0)
        c2_ref = k2.selectbox("Ref.No", cols2, index=min(1, len(cols2)-1))
        c2_noun = k3.selectbox("Noun", cols2, index=min(2, len(cols2)-1))
        c2_mod = k4.selectbox("Modifier", cols2, index=min(3, len(cols2)-1))
        c2_long = k5.selectbox("Long Description", cols2, index=min(4, len(cols2)-1))
        c2_uom = k6.selectbox("Uom", cols2, index=min(5, len(cols2)-1))

        if st.button("Ekstrak Atribut Massal untuk File 2", type="primary"):
            extracted_list = []
            prog2 = st.progress(0)
            st_txt2 = st.empty()
            tot2 = len(df_in2)

            for idx, r in df_in2.iterrows():
                noun_v = str(r[c2_noun]).strip().upper()
                mod_v = str(r[c2_mod]).strip().upper()
                long_v = str(r[c2_long]).strip()
                mc_v = str(r[c2_code]).strip()
                st_txt2.text(f"Mengekstrak {idx+1}/{tot2}: [{mc_v}] {noun_v} - {mod_v}...")

                schema_df = df_ms[(df_ms["NOUN"] == noun_v) & (df_ms["MODIFIER"] == mod_v)].sort_values("SEQ")
                if not schema_df.empty:
                    ext = extract_attributes(long_v, noun_v, mod_v, schema_df)
                    attr_map = {a.seq: a.value for a in ext.attributes}
                    std_long = ext.long_description
                    full_attrs = [
                        {
                            "seq": int(sr["SEQ"]),
                            "mandatory": sr["MANDATORY"],
                            "attribute": sr["ATTRIBUTE"],
                            "value": attr_map.get(int(sr["SEQ"]), ""),
                        }
                        for _, sr in schema_df.iterrows()
                    ]
                else:
                    full_attrs = []
                    std_long = long_v

                extracted_list.append({
                    "psc_code": default_psc,
                    "psc_material_code": mc_v,
                    "ref_no": str(r[c2_ref]).strip(),
                    "noun": noun_v,
                    "modifier": mod_v,
                    "uom": str(r[c2_uom]).strip().upper() or "EA",
                    "long_description": std_long,
                    "attributes": full_attrs,
                })
                prog2.progress((idx + 1) / tot2)
                time.sleep(0.2)

            st.session_state.file2_items = extracted_list
            st_txt2.success("Ekstraksi atribut Tahap 2 selesai!")

    if st.session_state.file2_items:
        for i, item in enumerate(st.session_state.file2_items):
            df_attr = pd.DataFrame(item["attributes"])
            if df_attr.empty:
                st.warning(f"{i+1}. [{item['psc_material_code']}] Skema {item['noun']} - {item['modifier']} tidak ditemukan.")
                continue
            missing_m = df_attr[(df_attr["mandatory"] == "M") & (df_attr["value"].astype(str).str.strip() == "")]
            lbl = f"⚠️ {len(missing_m)} Mandatory Kosong" if len(missing_m) > 0 else "✅ Lengkap"

            with st.expander(f"{i+1}. [{item['psc_material_code']}] {item['noun']} - {item['modifier']} | {lbl}", expanded=(len(missing_m) > 0)):
                item["long_description"] = st.text_area("Long Description (Baku)", item["long_description"], key=f"f2_ld_{i}")
                edited_df = st.data_editor(df_attr, use_container_width=True, disabled=["seq", "mandatory", "attribute"], key=f"f2_ed_{i}")
                item["attributes"] = edited_df.to_dict(orient="records")

        buf_f2 = build_file2_buffer(st.session_state.file2_items, df_ms)
        st.download_button(
            label="Download File 2 (Data_Extraction_Upload.xlsx)",
            data=buf_f2,
            file_name="Data_Extraction_Upload.xlsx",
            type="primary",
        )
