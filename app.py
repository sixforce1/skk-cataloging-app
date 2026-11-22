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

MS_FILE = "Material_Standard_26112014119179.xlsx"

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

    # Ekstrak seluruh 1.114 pasangan unik NOUN & MODIFIER langsung dari Material_Standard
    df_nm = (
        df_ms[["NOUN", "MODIFIER"]]
        .drop_duplicates()
        .sort_values(["NOUN", "MODIFIER"])
        .reset_index(drop=True)
    )
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

class BatchItemDecision(BaseModel):
    row_index: int = Field(description="Nomor urut baris input yang diproses")
    noun: str = Field(description="NOUN persis dari daftar resmi SKK Migas")
    modifier: str = Field(description="MODIFIER persis dari daftar resmi SKK Migas")
    confidence: int = Field(description="Skor keyakinan 0-100")
    reason: str = Field(description="Alasan singkat pemilihan Noun & Modifier")

class BatchClassificationResult(BaseModel):
    items: list[BatchItemDecision]

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

def filter_relevant_nm(texts: list[str], df_nm: pd.DataFrame) -> list[str]:
    nm_pairs = list(zip(df_nm["NOUN"], df_nm["MODIFIER"]))
    tokens = set()
    for t in texts:
        cleaned = (
            t.upper()
            .replace(",", " ")
            .replace(";", " ")
            .replace("/", " ")
            .replace("-", " ")
        )
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
        "Tentukan 3 kandidat (NOUN | MODIFIER) paling tepat dan SPESIFIK dari DAFTAR RESMI untuk material berikut.\n"
        "ATURAN WAJIB:\n"
        "1. PRIORITASKAN NOUN & MODIFIER SPESIFIK berdasarkan jenis barang utamanya (contoh: BEARING -> BEARING | BALL atau BEARING | ROLLER; "
        "TAPE -> TAPE | MASKING atau TAPE | ELECTRICAL atau TAPE | ADHESIVE; FUSE -> FUSE | --; GASKET -> GASKET | SPIRAL WOUND atau GASKET | RING JOINT atau GASKET | FLAT; "
        "FILTER -> FILTER | AIR atau FILTER | OIL; SWITCH -> SWITCH | ROCKER atau SWITCH | PRESSURE; GAUGE -> GAUGE | PRESSURE atau GAUGE | TEMPERATURE; HELMET -> HELMET | SAFETY).\n"
        "2. Jangan memilih 'SPARE PARTS | --' hanya karena ada nama merek/mesin (seperti SKF, NORD, CAT, SOLAR). "
        "Gunakan 'SPARE PARTS | --' HANYA sebagai pilihan terakhir jika jenis barangnya benar-benar tidak ada Noun standarnya di daftar resmi.\n"
        "3. NOUN dan MODIFIER wajib persis sama dengan DAFTAR RESMI.\n\n"
        f"DESKRIPSI MATERIAL MENTAH:\n\"{raw_text}\"\n\n"
        f"DAFTAR RESMI (NOUN | MODIFIER):\n{chr(10).join(filtered)}"
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
        "Tugas: Klasifikasikan setiap item pada DAFTAR INPUT ke pasangan (NOUN | MODIFIER) paling spesifik dari DAFTAR RESMI.\n"
        "ATURAN SANGAT PENTING:\n"
        "1. Selalu utamakan NOUN dan MODIFIER spesifik sesuai benda utamanya (misalnya BEARING -> BEARING | BALL / ROLLER; "
        "TAPE -> TAPE | MASKING / ADHESIVE / ELECTRICAL; FUSE -> FUSE | --; GASKET -> GASKET | FLAT / SPIRAL WOUND / RING JOINT; "
        "FILTER -> FILTER | AIR / OIL / FUEL; SWITCH -> SWITCH | ROCKER / PRESSURE; GAUGE -> GAUGE | PRESSURE; HELMET -> HELMET | SAFETY; TUBING -> TUBING | ...).\n"
        "2. DILARANG memilih 'SPARE PARTS | --' apabila benda tersebut memiliki Noun standar di daftar resmi, meskipun deskripsinya menyebut merek/unit mesin (SKF, NORD, CAT, SOLAR, dll).\n"
        "3. Pastikan NOUN dan MODIFIER tertulis PERSIS seperti di DAFTAR RESMI.\n\n"
        f"DAFTAR INPUT MATERIAL:\n{items_block}\n\n"
        f"DAFTAR RESMI (NOUN | MODIFIER):\n{chr(10).join(filtered)}"
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
        "Anda adalah cataloguer senior SKK Migas. Ekstrak nilai spesifikasi dari deskripsi mentah ke slot atribut baku berikut.\n"
        "Aturan ketat:\n"
        "1. Gunakan HURUF KAPITAL (UPPERCASE) untuk semua nilai.\n"
        "2. Jangan mengarang spesifikasi yang tidak tertulis di deskripsi mentah; isi string kosong \"\" jika tidak ditemukan.\n"
        "3. Hilangkan spasi berlebih atau karakter tab.\n"
        "4. Susun 'short_description' maksimal 40 karakter dan 'long_description' mengikuti urutan atribut terisi.\n\n"
        f"KLASIFIKASI TERPILIH: NOUN = {noun}, MODIFIER = {modifier}\n"
        f"DESKRIPSI MENTAH: \"{raw_text}\"\n\n"
        f"SKEMA ATRIBUT RESMI SKK MIGAS:\n{chr(10).join(schema_lines)}"
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
        [
            "INTEGER", "ALPHANUMERIC", "ALPHANUMERIC", "ALPHANUMERIC",
            "ALPHANUMERIC", "ALPHANUMERIC", "ALPHANUMERIC", "HEADER DATA", "DATA TYPE"
        ],
        [
            "No. of the material data ", "The Code/KEYMAP# of the material",
            "NOUN as per the CLASS master", "MODIFIER as per the CLASS master",
            "Short description of the material", "Long description of the material",
            "Uom of the material", None, "DESCRIPTION"
        ],
        ["2", "20", "30", "30", "100", "3000", "10", None, "SIZE"],
        [
            "No.", "PSC Material code", "Noun", "Modifier",
            "Material - Short Description", "Material - Long Description",
            "Uom", None, "HEADER"
        ],
        [
            1, "100001", "SPARE PARTS", "--", "PLATE A 5mm",
            "CONTACTOR:AUX,50/60HZ,3NO 2NC; MANUFACTURER: KONECRANES; P/N: CAD32V7",
            "KGS", None, "EXAMPLE 1"
        ],
        [
            2, "100003", "SPARE PARTS", "--", "PLATE A 6mm",
            "BRAKE; MANUFACTURER: KONECRANES; P/N: NM38720NR2V",
            "KGS", None, "EXAMPLE 2"
        ],
        [
            3, "100004", "SPARE PARTS", "--", "PLATE A 7mm",
            "BRAKE; MANUFACTURER: KONECRANES; P/N: NM38730NR2V",
            "KGS", None, "EXAMPLE 3"
        ],
        [
            4, "100005", "SPARE PARTS", "--", "PLATE A 8mm",
            "BRAKE:; MANUFACTURER: KONECRANES; P/N: NM38740NR2V",
            "KGS", None, "EXAMPLE 4"
        ],
        [
            5, "100006", "SPARE PARTS", "--", "PLATE A 9mm",
            "SET:BRAKE DISC; MANUFACTURER: KONECRANES; P/N: SP-NM38730JPMV",
            "KGS", None, "EXAMPLE 5"
        ],
    ]
    for r in format_rows:
        ws2.append(r)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf

# --- 5. Generator File 2 (Data_Extraction_Upload) ---
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

# --- 6. Antarmuka Utama Aplikasi dengan Navigasi Menu ---
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
    st.header("Pengaturan Default")
    default_psc = st.text_input("PSC Code", value=st.secrets.get("PSC_CODE", "A095"))
    st.success(
        f"Master Tunggal Aktif:\n"
        f"- File: {MS_FILE}\n"
        f"- {len(df_nm):,} Noun-Modifier\n"
        f"- {len(df_ms):,} Baris Atribut"
    )

# ==============================================================================
# MENU 1: ANALISA & EKSTRAK SATUAN
# ==============================================================================
if menu_pilihan == "Menu 1: Analisa & Ekstrak Satuan":
    st.title("Menu 1: Analisa & Ekstrak Satuan")
    st.caption("Klasifikasi dan ekstraksi atribut lengkap untuk 1 item material secara cepat.")

    t1_in, t1_rev = st.tabs(["1. Analisa & Klasifikasi Material", "2. Review & Download Hasil Satuan"])

    with t1_in:
        c1, c2, c3 = st.columns([2, 2, 1])
        mat_code = c1.text_input("PSC Material Code (Wajib)", placeholder="Contoh: 51068322")
        ref_no = c2.text_input("Ref.No", placeholder="Contoh: A095_1367818")
        uom = c3.text_input("UoM", value="EA")
        raw_desc = st.text_area(
            "Deskripsi Material Mentah (Short Text + Long Text dari user / vendor / SAP):",
            placeholder="Tempel deskripsi lengkap material di sini...",
            height=120,
        )

        if st.button("Cari Rekomendasi Noun - Modifier", type="primary") and raw_desc.strip():
            with st.spinner("AI sedang mencocokkan dengan standar SKK Migas..."):
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
            st.subheader("Rekomendasi AI (Pilih yang Paling Sesuai)")
            options = [
                f"{c.noun} | {c.modifier}  —  (Keyakinan: {c.confidence}% | {c.reason})"
                for c in res.top_candidates
            ]
            options.append(f"Pilih Manual dari {len(df_nm):,} Daftar Lengkap...")
            choice = st.radio("Kandidat Klasifikasi:", options, key="rad_m1")

            if choice.startswith("Pilih Manual"):
                all_nm = [f"{r['NOUN']} | {r['MODIFIER']}" for _, r in df_nm.iterrows()]
                manual_pick = st.selectbox("Ketik untuk mencari Noun | Modifier:", all_nm, key="sel_m1")
                sel_noun, sel_mod = [x.strip() for x in manual_pick.split("|")]
            else:
                sel_noun, sel_mod = [x.strip() for x in choice.split("—")[0].split("|")]

            if st.button("Ekstrak Atribut & Simpan ke Daftar"):
                schema_df = df_ms[(df_ms["NOUN"] == sel_noun) & (df_ms["MODIFIER"] == sel_mod)].sort_values("SEQ")
                with st.spinner(f"Menyusun atribut baku untuk {sel_noun} - {sel_mod}..."):
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
                    st.success("Tersimpan! Buka tab '2. Review & Download Hasil Satuan' untuk memeriksa atributnya.")

    with t1_rev:
        if not st.session_state.single_items:
            st.info("Belum ada material yang diproses di Menu 1.")
        else:
            for i, item in enumerate(st.session_state.single_items):
                df_attr = pd.DataFrame(item["attributes"])
                missing_m = df_attr[(df_attr["mandatory"] == "M") & (df_attr["value"].astype(str).str.strip() == "")]
                status_label = f"⚠️ {len(missing_m)} Atribut Mandatory (M) Kosong" if len(missing_m) > 0 else "✅ Lengkap"

                with st.expander(f"{i+1}. [{item['psc_material_code'] or 'NO-CODE'}] {item['noun']} - {item['modifier']} | {status_label}", expanded=True):
                    c_a, c_b, c_c = st.columns(3)
                    item["psc_material_code"] = c_a.text_input("PSC Material Code", item["psc_material_code"], key=f"m1_mc_{i}")
                    item["ref_no"] = c_b.text_input("Ref.No", item["ref_no"], key=f"m1_rn_{i}")
                    item["uom"] = c_c.text_input("UoM", item["uom"], key=f"m1_um_{i}")
                    item["short_description"] = st.text_input("Material - Short Description (Baku)", item["short_description"], key=f"m1_sd_{i}")
                    item["long_description"] = st.text_area("Material - Long Description (Baku)", item["long_description"], key=f"m1_ld_{i}")

                    edited_df = st.data_editor(
                        df_attr,
                        use_container_width=True,
                        disabled=["seq", "mandatory", "attribute"],
                        key=f"m1_ed_{i}",
                    )
                    item["attributes"] = edited_df.to_dict(orient="records")

            st.divider()
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
            buf1_s = build_file1_buffer(df_s_f1)
            buf2_s = build_file2_buffer(st.session_state.single_items, df_ms)

            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button(
                label="Download File 1 (Source_Data_Upload.xlsx)",
                data=buf1_s,
                file_name="Source_Data_Upload.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
            col_dl2.download_button(
                label="Download File 2 (Data_Extraction_Upload.xlsx)",
                data=buf2_s,
                file_name="Data_Extraction_Upload.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

# ==============================================================================
# MENU 2: BATCH TAHAP 1 (SOURCE_DATA_UPLOAD)
# ==============================================================================
elif menu_pilihan == "Menu 2: Batch Tahap 1 (Source_Data_Upload)":
    st.title("Menu 2: Batch Upload Tahap 1 (Source_Data_Upload)")
    st.caption("Klasifikasi massal cepat (Batch 10 item per request) dengan prioritas Noun & Modifier spesifik dari master SKK Migas.")

    sample_df = pd.DataFrame([
        {
            "PSC Material Code": "51068322",
            "Material - Short Description": "GAS, CALIBRATION, GC CYLINDER 205",
            "Material - Long Description": "GAS, CALIBRATION, METHANE 50%, NITROGEN BALANCE, GC CYLINDER 205, 2000 PSI",
            "Uom": "BT",
        }
    ])
    buf_tpl = io.BytesIO()
    with pd.ExcelWriter(buf_tpl, engine="openpyxl") as w:
        sample_df.to_excel(w, index=False, sheet_name="Raw_Material")
    buf_tpl.seek(0)
    st.download_button("Download Contoh Template Input Mentah (.xlsx)", buf_tpl, "Template_Input_Mentah.xlsx")

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
            results = []
            prog = st.progress(0)
            status_txt = st.empty()
            total = len(df_in)
            batch_size = 10
            valid_pairs = set(zip(df_nm["NOUN"], df_nm["MODIFIER"]))

            for start_idx in range(0, total, batch_size):
                end_idx = min(start_idx + batch_size, total)
                status_txt.text(f"Memproses batch baris {start_idx+1} s/d {end
