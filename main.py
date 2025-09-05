import os
import re
import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv
from openai import AzureOpenAI


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


st.title("Cargo Container OCR")
st.write("Upload cargo container images to extract and clean container IDs.")

uploaded_files = st.file_uploader(
    "Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

reader=easyocr.Reader(['en'])

prompt = """You are given OCR extracted text from cargo containers.
Your job is to clean the text and return only the valid container ID in valid format.
Look for this pattern :- It is 4 letters combined with 6 digits.
Do not combine digits from different detecions only use them if u find 6 digits together.
If brackets or other charecters get detected remove them to check the 4 letter and 6 digit cimbination can be made.
Refer to the given examples.
Examples:
OCR: "lESU 425918 42G1 MeY %0 5 88 Qui 28"
Output: "LESU425918"

OCR: "MAEU 123456 7"
Output: "MAEU1234567"

OCR: "CMA CGM U 987654 3"
Output: "CMAU9876543"

OCR: "PSSU 4261 27,150 KG [403377"
Output: "PSSU403377"

Now clean the following OCR text:"""


def regex_extract(ocr_text: str):
    match = re.search(r"\b([A-Z]{4}\d{7})\b", ocr_text.replace(" ", "").upper())
    return match.group(1) if match else None

def clean_with_llm(ocr_text: str, prompt: str) -> str:
    
    regex_result = regex_extract(ocr_text)
    if regex_result:
        return regex_result

    if not client:
        return "⚠️ Azure OpenAI not configured. Showing raw OCR text."
    try:
        resp = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts and cleans cargo container IDs."},
                {"role": "user", "content": f"{prompt}\n\nOCR extracted: {ocr_text}\nPlease return the cleaned container ID only."}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error calling Azure OpenAI: {e}"


# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         image = Image.open(uploaded_file).convert("RGB")
#         results = reader.readtext(np.array(image))

        
#         annotated_img = np.array(image).copy()
#         for (bbox, text, confidence) in results:
#             pts = np.array(bbox).astype(int).reshape((-1, 1, 2))
#             cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#         st.image(annotated_img, caption="Detected text regions", use_column_width=True)

#         st.subheader("OCR Results")
#         all_texts = []
#         for (bbox, text, confidence) in results:
#             st.write(f"**{text}** (confidence: {confidence:.2f})")
#             # if confidence > 0.4:
#             all_texts.append(text)

#         if all_texts:
#             raw_text = " ".join(all_texts)
#             st.subheader("✅ Cleaned Container ID")
#             cleaned = clean_with_llm(raw_text, prompt)
#             st.success(cleaned)
# else:
#     st.info("Upload one or more images to start OCR processing -->")
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        upscaled_img = cv2.resize(img_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY)

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        sharpened = cv2.medianBlur(sharpened, 3)

        results = reader.readtext(gray)

        annotated_img = upscaled_img.copy()
        all_texts = []
        for (bbox, text, confidence) in results:
            pts = np.array(bbox).astype(int).reshape((-1, 1, 2))
            cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            all_texts.append(text)

        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption="Sharpened & Detected text regions", use_column_width=True)

        raw_text = " ".join(all_texts)
        cleaned = clean_with_llm(raw_text, prompt)
        st.subheader("Cleaned Container ID")
        st.success(cleaned)

else:
    st.info("Upload one or more images to start OCR processing -->")
