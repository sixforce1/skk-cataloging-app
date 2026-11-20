import io
import os
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

st.set_page_config(page_title="SKK Migas Auto-Cataloging", layout="wide")

NM_FILE = "Noun_Modifier_List_261120141113553.xlsx"
MS_FILE = "Material_Standard_26112014119179.xlsx"

# --- 1. Pembaca Master Excel SKK Migas (Cache di Memori Server) ---
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
    reason: str = Field(description="Alasan singkat pemilihan klasifikasi dalam Bahasa Indonesia")

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

def classify_material(raw_description: str, df_nm: pd.DataFrame) -> ClassificationResult:
    client = get_gemini_client()
    nm_pairs = list(zip(df_nm["NOUN"], df_nm["MODIFIER"]))
    words = {w.strip(",.;:-/()") for w in raw_description.upper().split() if len(w) > 2}
    filtered = [f"{n} | {m}" for n, m in nm_pairs if any(w in n or w in m for w in words)]
    if len(filtered) < 25:
        filtered = [f"{n} | {m}" for n, m in nm_pairs]

    prompt = f"""
Anda adalah ahli katalogisasi material migas SKK Migas.
Tugas: Tentukan 3 kandidat (NOUN | MODIFIER) paling tepat dari DAFTAR RESMI di bawah untuk deskripsi material berikut.
Jika barang merupakan komponen spesifik mesin/OEM tanpa bentuk standar umum, sertakan 'SPARE PARTS | --' sebagai kandidat.
Jangan pernah mengarang NOUN atau MODIFIER di luar daftar resmi.

DESKRIPSI MATERIAL MENTAH:
"{raw_description}"

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

def extract_attributes(raw_description: str, noun: str, modifier: str, schema_df: pd.DataFrame) -> ExtractionResult:
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
DESKRIPSI MENTAH: "{raw_description}"

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

# --- 4. Pembuat File Excel (File 1 & File 2) di Memori ---
def build_excel_buffers(items: list[dict], df_ms: pd.DataFrame):
    rows_file1 = []
    rows_file2 = []
    seq_counter_f2 = 1

    for idx, item in enumerate(items, start=1):
        psc_code = str(item["psc_code"]).strip()
        mat_code = str(item["psc_material_code"]).strip()
        ref_no = str(item["ref_no"]).strip().replace("\t", "")
        noun = str(item["noun"]).strip().upper()
        modifier = str(item["modifier"]).strip().upper()
        uom = str(item["uom"]).strip().upper()
        short_desc = str(item["short_description"]).strip().upper()[:100]
        long_desc = str(item["long_description"]).strip().upper()[:3000]

        rows_file1.append({
            "No.": idx,
            "PSC Material Code": mat_code,
            "Noun": noun,
            "Modifier": modifier,
            "Material - Short Description": short_desc,
            "Material - Long Description": long_desc,
            "Uom": uom,
        })

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

    buf1 = io.BytesIO()
    with pd.ExcelWriter(buf1, engine="openpyxl") as writer:
        pd.DataFrame(rows_file1).to_excel(writer, sheet_name="Source_Data_Upload", index=False)
    buf1.seek(0)

    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
        pd.DataFrame(rows_file2).to_excel(writer, sheet_name="Data_Extraction_Upload", index=False)
    buf2.seek(0)

    return buf1, buf2

# --- 5. Antarmuka Utama Aplikasi ---
st.title("SKK Migas Auto-Cataloging System")
st.caption("Klasifikasi Otomatis Noun-Modifier & Generator File Upload Tahap 1 dan Tahap 2")

df_nm, df_ms = load_master_data()

if "processed_items" not in st.session_state:
    st.session_state.processed_items = []

with st.sidebar:
    st.header("Pengaturan Default")
    default_psc = st.text_input("PSC Code", value=st.secrets.get("PSC_CODE", "A095"))
    st.success(f"Master SKK Migas aktif:\n- {len(df_nm):,} Noun-Modifier\n- {len(df_ms):,} Baris Atribut")

tab1, tab2 = st.tabs(["1. Analisa & Klasifikasi Material", "2. Review & Download File 1 & 2"])

with tab1:
    c1, c2, c3 = st.columns([2, 2, 1])
    mat_code = c1.text_input("PSC Material Code", placeholder="Contoh: 51072464")
    ref_no = c2.text_input("Ref.No", placeholder="Contoh: A095_1367818")
    uom = c3.text_input("UoM", value="EA")
    raw_desc = st.text_area(
        "Deskripsi Material Mentah (ketik apa adanya dari user / vendor / SAP):",
        placeholder="Contoh: BONNET NEEDLE VALVE 1/4 IN TUBE FITTING 316SS PN SS-1RS4 SWAGELOK",
        height=100,
    )

    if st.button("Cari Rekomendasi Noun - Modifier", type="primary") and raw_desc.strip():
        with st.spinner("AI sedang mencocokkan dengan standar SKK Migas..."):
            st.session_state.last_class = classify_material(raw_desc, df_nm)
            st.session_state.last_input = {
                "psc_code": default_psc,
                "psc_material_code": mat_code,
                "ref_no": ref_no,
                "uom": uom,
                "raw_desc": raw_desc,
            }

    if "last_class" in st.session_state:
        res = st.session_state.last_class
        st.subheader("Rekomendasi AI (Pilih yang Paling Sesuai)")
        options = [
            f"{c.noun} | {c.modifier}  —  (Keyakinan: {c.confidence}% | {c.reason})"
            for c in res.top_candidates
        ]
        options.append("Pilih Manual dari 1.114 Daftar Lengkap...")
        choice = st.radio("Kandidat Klasifikasi:", options)

        if choice.startswith("Pilih Manual"):
            all_nm = [f"{r['NOUN']} | {r['MODIFIER']}" for _, r in df_nm.iterrows()]
            manual_pick = st.selectbox("Ketik untuk mencari Noun | Modifier:", all_nm)
            sel_noun, sel_mod = [x.strip() for x in manual_pick.split("|")]
        else:
            sel_noun, sel_mod = [x.strip() for x in choice.split("—")[0].split("|")]

        if st.button("Ekstrak Atribut & Simpan ke Daftar Upload"):
            schema_df = df_ms[(df_ms["NOUN"] == sel_noun) & (df_ms["MODIFIER"] == sel_mod)].sort_values("SEQ")
            with st.spinner(f"Menyusun atribut baku untuk {sel_noun} - {sel_mod}..."):
                inp = st.session_state.last_input
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
                st.session_state.processed_items.append({
                    "psc_code": inp["psc_code"],
                    "psc_material_code": inp["psc_material_code"],
                    "ref_no": inp["ref_no"],
                    "noun": sel_noun,
                    "modifier": sel_mod,
                    "uom": inp["uom"],
                    "short_description": ext.short_description,
                    "long_description": ext.long_description,
                    "attributes": full_attrs,
                })
                st.success(f"Tersimpan! Buka tab '2. Review & Download File 1 & 2' untuk memeriksa dan mengunduh.")

with tab2:
    if not st.session_state.processed_items:
        st.info("Belum ada material yang diproses. Silakan mulai dari Tab 1.")
    else:
        for i, item in enumerate(st.session_state.processed_items):
            df_attr = pd.DataFrame(item["attributes"])
            missing_m = df_attr[(df_attr["mandatory"] == "M") & (df_attr["value"].astype(str).str.strip() == "")]
            status_label = f"⚠️ {len(missing_m)} Atribut Mandatory (M) Kosong" if len(missing_m) > 0 else "✅ Lengkap"

            with st.expander(f"{i+1}. [{item['psc_material_code'] or 'NO-CODE'}] {item['noun']} - {item['modifier']} | {status_label}", expanded=True):
                c_a, c_b, c_c = st.columns(3)
                item["psc_material_code"] = c_a.text_input("PSC Material Code", item["psc_material_code"], key=f"mc_{i}")
                item["ref_no"] = c_b.text_input("Ref.No", item["ref_no"], key=f"rn_{i}")
                item["uom"] = c_c.text_input("UoM", item["uom"], key=f"um_{i}")
                item["short_description"] = st.text_input("Material - Short Description", item["short_description"], key=f"sd_{i}")
                item["long_description"] = st.text_area("Material - Long Description", item["long_description"], key=f"ld_{i}")

                edited_df = st.data_editor(
                    df_attr,
                    use_container_width=True,
                    disabled=["seq", "mandatory", "attribute"],
                    key=f"ed_{i}",
                )
                item["attributes"] = edited_df.to_dict(orient="records")

        st.divider()
        buf1, buf2 = build_excel_buffers(st.session_state.processed_items, df_ms)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button(
            label="Download File 1 (Source_Data_Upload.xlsx)",
            data=buf1,
            file_name="Source_Data_Upload.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
        col_dl2.download_button(
            label="Download File 2 (Data_Extraction_Upload.xlsx)",
            data=buf2,
            file_name="Data_Extraction_Upload.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
