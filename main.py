import streamlit as st
import cv2
import numpy as np
from PIL import Image
import re
from functools import lru_cache
from dotenv import load_dotenv
from openai import AzureOpenAI
import easyocr
import os

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-nano")
deployment_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

if api_key and endpoint:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=deployment_api_version,
        azure_endpoint=endpoint,
    )
else:
    client = None
reader=easyocr.Reader(['en'])

st.title("Cargo Container OCR")
st.write("Upload cargo container images to extract and clean container IDs.")

uploaded_files = st.file_uploader(
    "Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)



def filter_tokens(results):
    unwanted = ["TARE", "GROSS", "NET", "KG", "LBS", "CUFT", "CU", "CAP"]
    tokens = []
    for bbox, text, conf in results:
        txt = text.strip().upper()
        if any(u in txt for u in unwanted):
            continue
        tokens.append({"text": txt, "conf": conf, "bbox": bbox})
    return tokens

def bbox_extract_container_id(tokens, y_tolerance=15):
    letters = [t for t in tokens if re.fullmatch(r"[A-Z]{4}", t["text"])]
    digits = [t for t in tokens if re.fullmatch(r"\d{6}", t["text"])]

    if not digits:
        digits = [t for t in tokens if re.fullmatch(r"\d{4,6}", t["text"])]

    for l in letters:
        ly = np.mean([p[1] for p in l["bbox"]])
        for d in digits:
            dy = np.mean([p[1] for p in d["bbox"]])
            if abs(ly - dy) <= y_tolerance:
                return l["text"] + d["text"]
    return None

@lru_cache(maxsize=1000)
def llm_extract_container_id_hashable(tokens_tuple, client, deployment_name):
    if not client:
        return " No LLM client available"

    tokens = [{"text": t, "conf": c, "bbox": eval(bbox_str)} for t, c, bbox_str in tokens_tuple]

    token_texts = [t["text"] for t in tokens]

    examples = [
        {"tokens": ["PSOU","514277"], "id": "PSOU514277"},
        {"tokens": ["CBCU","201984"], "id": "CBCU201984"},
        {"tokens": ["GNVV","075555"], "id": "GNVV075555"}
    ]
    example_text = "\n".join([f"Tokens: {ex['tokens']} -> Container ID: {ex['id']}" for ex in examples])

    prompt = (
        f"You are an assistant that extracts cargo container IDs from OCR tokens.\n"
        f"Rules:\n"
        f"1. Container ID = 4 uppercase letters + exactly 6 digits.\n"
        f"2. Use only the tokens provided.\n"
        f"3. Ignore unrelated numbers like weights, tare, gross, etc.\n"
        f"Output only the container ID.\n\n"
        f"Examples:\n{example_text}\n\n"
        f"Now process the following tokens:\n{token_texts}\n\n"
        f"Container ID:"
    )

    try:
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role":"system","content":"You extract container IDs."},
                      {"role":"user","content":prompt}],
            temperature=0
        )
        result = resp.choices[0].message.content.strip()
        if re.fullmatch(r"[A-Z]{4}\d{6}", result):
            return result
        else:
            return f" Invalid LLM output: {result}"
    except Exception as e:
        return f" LLM error: {e}"
def bbox_extract_container_id(tokens):
    valid_tokens = [t for t in tokens if isinstance(t["bbox"], (list, np.ndarray)) and len(t["bbox"]) == 4]
    if not valid_tokens:
        return None

    letters = [t for t in valid_tokens if re.fullmatch(r"[A-Z]{4}", t["text"])]
    digits  = [t for t in valid_tokens if re.fullmatch(r"\d{6}", t["text"])]
    if not letters or not digits:
        return None

    letters_sorted = sorted(letters, key=lambda l: np.mean([p[1] for p in l["bbox"]]))
    digits_sorted  = sorted(digits,  key=lambda d: np.mean([p[1] for p in d["bbox"]]))

    for l in letters_sorted:
        ly = np.mean([p[1] for p in l["bbox"]])
        for d in digits_sorted:
            dy = np.mean([p[1] for p in d["bbox"]])
            if abs(ly - dy) < 20: 
                return l["text"] + d["text"]

    return None


if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        results = reader.readtext(np.array(image))

        annotated_img = np.array(image).copy()
        for bbox, text, conf in results:
            pts = np.array(bbox).astype(int).reshape((-1,1,2))
            cv2.polylines(annotated_img, [pts], isClosed=True, color=(0,255,0), thickness=2)

        st.image(annotated_img, caption="Detected text regions", use_container_width=True)

        st.subheader("OCR Results")
        for bbox, text, confidence in results:
            st.write(f"**{text}** (confidence: {confidence:.2f})")

        if results:
            st.subheader("Cleaned Container ID")
            
            tokens = []
            for bbox, text, conf in results:
                cleaned_text = text.replace("[","").replace("]","").replace("|","").replace("'","").strip()
                if cleaned_text: 
                    tokens.append({"text": cleaned_text, "conf": conf, "bbox": bbox})

            container_id = bbox_extract_container_id(tokens)

            if not container_id:
                tokens_tuple = tuple((t["text"], t["conf"], str(t["bbox"])) for t in tokens)
                container_id = llm_extract_container_id_hashable(tokens_tuple, client, deployment_name)

            st.success(container_id or "Container ID not found")
else:
    st.info("Upload one or more images to start OCR processing -->")
