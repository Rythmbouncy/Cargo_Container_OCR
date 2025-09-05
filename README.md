# Cargo Container OCR

A Streamlit-based application to extract and clean cargo container IDs from images using EasyOCR and optional Azure OpenAI for advanced cleaning. The app applies **image upscaling and sharpening** to improve OCR accuracy and ensures the container IDs follow the standard format.  

---

## Features

- Upload one or more images of cargo containers.
- Automatic image upscaling and sharpening for better OCR accuracy.
- OCR text extraction using **EasyOCR**.
- Intelligent post-processing to clean container IDs:
  - Regex-based extraction of 4-letter + 6-7 digit container IDs.
  - Optional fuzzy corrections for common OCR mistakes (`N↔H`, `0↔O`, `1↔I`).
- Optional Azure OpenAI LLM integration to clean IDs in complex cases.
- Annotated display of detected text regions for visual confirmation.
- Cleaned container ID output displayed in the Streamlit app.

---

## Sample

<img src="app/Screenshot 2025-09-05 at 4.35.29 PM.png" alt="Alt text" width="400"/>

<img src="app/Screenshot 2025-09-05 at 4.36.23 PM.png" alt="Alt text" width="400"/>


## Installation

1. Clone the repository:

bash
git clone https://github.com/your-username/cargo-container-ocr.git
cd cargo-container-ocr

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies:

pip install -r requirements.txt

4. (Optional) Set up Azure OpenAI credentials in a .env file:

AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-nano
AZURE_OPENAI_API_VERSION=2025-01-01-preview

## Usage

bash
streamlit run app.py

1. Upload one or more cargo container images.

2. View detected text regions in the image.

3. See raw OCR results and cleaned container IDs.

4. Optional: Azure OpenAI will further clean IDs if regex-based extraction fails.