import io
import os
import time
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import openpyxl
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

st.set_page_config(page_title="SKK Migas Auto-Cataloging System", layout="wide")

NM_FILE = "Noun_Modifier_List_261120141113553.xlsx"
MS_FILE = "Material_Standard_26112014119179.xlsx"

# --- 1. Pembaca Master Excel SKK Migas ---
def read_skk_xlsx(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(filepath)
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
    df_nm = read_skk_xlsx(NM_FILE)
    df_nm.columns = [c.strip().upper() for c in df_nm.columns]
    df_nm["NOUN"] = df_nm["NOUN"].astype(str).str.strip().str.upper()
    df_nm["MODIFIER"] = df_nm["MODIFIER"].astype(str).str.strip().str.upper()

    df_ms = read_skk_xlsx(MS_FILE)
    df_ms.columns = [c.strip().upper() for c in df_ms.columns]
    df_ms["NOUN"] = df_ms["NOUN"].astype(str).str.strip().str.upper()
    df_ms["MODIFIER"] = df_ms["MODIFIER"].astype(str).str.strip().str.upper()
    df_ms["SEQ"] = df_ms["SEQ"].astype(int)
    df_ms["MANDATORY"] = df_ms["MANDATORY"].astype(str).str.strip().str.upper()
    df_ms["ATTRIBUTE"] = df_ms["ATTRIBUTE"].astype(str).str.strip().str.upper()
    return df_nm, df_ms

# --- 2. Skema Output Terstruktur AI ---
class CandidateNM(BaseModel):
    noun: str = Field(description="NOUN persis dari daftar resmi SKK Migas")
    modifier: str = Field(description="MODIFIER persis dari daftar resmi SKK Migas (atau '--')")
    confidence: int = Field(description="Skor keyakinan 0-100")
    reason: str = Field(description="Alasan singkat dalam Bahasa Indonesia")

class ClassificationResult(BaseModel):
    top_candidates: list[CandidateNM] = Field(description="3 kandidat Noun-Modifier terbaik")
    needs_review: bool = Field(description="True jika ambigu atau skor tertinggi < 85")

class ExtractedAttr(BaseModel):
    seq: int
    attribute: str
    value: str = Field(description="Nilai teknis UPPERCASE. Kosongkan ('') jika tidak ada di teks sumber.")

class ExtractionResult(BaseModel):
    attributes: list[ExtractedAttr]
    short_description: str = Field(description="Maksimal 40 karakter, UPPERCASE")
    long_description: str = Field(description="Deskripsi lengkap baku UPPERCASE sesuai urutan atribut")

# --- 3. Fungsi Panggilan Gemini API ---
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not api_key:
        st.error("GEMINI_API_KEY belum diatur di Streamlit Secrets!")
        st.stop()
    return genai.Client(api_key=api_key)

def get_model_name():
    return st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-4-hydrogen"))

def classify_material(raw_text: str, df_nm: pd.DataFrame) -> ClassificationResult:
    client = get_gemini_client()
    nm_pairs = list(zip(df_nm["NOUN"], df_nm["MODIFIER"]))
    words = {w.strip(",.;:-/()") for w in raw_text.upper().split() if len(w) > 2}
    filtered = [f"{n} | {m}" for n, m in nm_pairs if any(w in n or w in m for w in words)]
    if len(filtered) < 25:
        filtered = [f"{n} | {m}" for n, m in nm_pairs]

    prompt = f"""
Anda adalah ahli katalogisasi material migas SKK Migas.
Tugas: Tentukan 3 kandidat (NOUN | MODIFIER) paling tepat dari DAFTAR RESMI di bawah untuk deskripsi material berikut.
Jika barang merupakan komponen spesifik mesin/OEM tanpa bentuk standar umum, sertakan 'SPARE PARTS | --' sebagai kandidat.
Jangan pernah mengarang NOUN atau MODIFIER di luar daftar resmi.

DESKRIPSI MATERIAL MENTAH:
"{raw_text}"

DAFTAR RESMI (NOUN | MODIFIER):
{chr(10).join(filtered)}
"""
    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificationResult,
            temperature=0.1,
        ),
    )
    return ClassificationResult.model_validate_json(response.text)

def extract_attributes(raw_text: str, noun: str, modifier: str, schema_df: pd.DataFrame) -> ExtractionResult:
    client = get_gemini_client()
    schema_lines = [
        f"SEQ {r['SEQ']} | Mandatory: {r['MANDATORY']} | Attribute: {r['ATTRIBUTE']}"
        for _, r in schema_df.iterrows()
    ]

    prompt = f"""
Anda adalah cataloguer senior SKK Migas. Ekstrak nilai spesifikasi dari deskripsi mentah ke slot atribut baku berikut.
Aturan ketat:
1. Gunakan HURUF KAPITAL (UPPERCASE) untuk semua nilai.
2. Jangan mengarang spesifikasi yang tidak tertulis di deskripsi mentah; isi string kosong "" jika tidak ditemukan.
3. Hilangkan spasi berlebih atau karakter tab.
4. Susun 'short_description' maksimal 40 karakter dan 'long_description' mengikuti urutan atribut terisi.

KLASIFIKASI TERPILIH: NOUN = {noun}, MODIFIER = {modifier}
DESKRIPSI MENTAH: "{raw_text}"

SKEMA ATRIBUT RESMI SKK MIGAS:
{chr(10).join(schema_lines)}
"""
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

# --- 4. Generator File 1 (Sesuai Template Resmi Source_Data_Upload) ---
def build_file1_buffer(df_rows: pd.DataFrame) -> io.BytesIO:
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Source_Data_Upload"

    headers = [
        "No.",
        "PSC Material Code",
        "Noun",
        "Modifier",
        "Material - Short Description",
        "Material - Long Description",
        "Uom",
    ]
    ws1.append(headers)

    for idx, row in df_rows.reset_index(drop=True).iterrows():
        mat_code = str(row.get("PSC Material Code", "")).strip().replace("\t", "")[:20]
        noun = str(row.get("Noun", "")).strip().upper()[:30]
        modifier = str(row.get("Modifier", "")).strip().upper()[:30]
        short_desc = str(row.get("Material - Short Description", "")).strip().replace("\t", "")[:100]
        long_desc = str(row.get("Material - Long Description", "")).strip().replace("\t", "")[:3000]
        uom = str(row.get("Uom", "")).strip().upper()[:10]

        ws1.append([idx + 1, mat_code, noun, modifier, short_desc, long_desc, uom])

    # Sheet 2: Source_Data_Upload_Format (Persis seperti template resmi SKK Migas)
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
        [5, "100006", "SPARE PARTS", "--", "PLATE A 9mm", "SET:BRAKE DISC; MANUFACTURER: KONECRANES; P/N: SP-
