# Invoice PII Redaction Tool - Streamlit App

A web-based application for automatically detecting and redacting personally identifiable information (PII) from invoices using Streamlit.

## Features

- 🔒 **Automatic PII Detection**: Names, addresses, phone numbers, emails, bank accounts
- 📄 **Multi-format Support**: Images (PNG, JPG, JPEG, TIFF, BMP) and PDFs
- 🎯 **Smart Business Data Preservation**: Avoids redacting legitimate business information
- 📊 **Visual Results**: Side-by-side comparison with download capability
- 📈 **Statistics Dashboard**: Detailed breakdown of detected PII types

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
- poppler-utils (for PDF processing)

### Installation

1. **Install system dependencies:**

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr poppler-utils
   ```

   **macOS:**
   ```bash
   brew install tesseract poppler
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

**Note:** The app uses `opencv-python-headless` to avoid GUI dependency issues in server environments.

### Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. **Upload** an invoice file (drag & drop or click to browse)
2. **Configure** processing options:
   - For PDFs: Toggle "Force OCR" if needed
3. **Review** the processing results:
   - View statistics of detected PII
   - Compare original vs redacted versions
4. **Download** the redacted files

## Technical Details

### PII Detection Methods

- **Names**: spaCy Named Entity Recognition (NER)
- **Addresses**: Custom spatial grouping algorithm for US addresses
- **Phone Numbers**: Regex patterns for US phone formats
- **Emails**: Standard email regex validation
- **Bank Accounts**: Pattern matching for various account number formats

### PDF Processing

- **Native Text Extraction**: Uses PyMuPDF for text-based PDFs
- **OCR Fallback**: Converts to images and uses Tesseract OCR
- **Hybrid Approach**: Automatically falls back to OCR if native extraction yields poor results

### Business Data Preservation

The tool intelligently preserves legitimate business information:
- Monetary values and prices
- Quantities and measurements
- Business terms and service descriptions
- Invoice numbers and dates
- Tax rates and percentages

## File Structure

```
├── streamlit_app.py              # Main Streamlit application
├── invoice_redaction_clean.py    # Core redaction logic
├── invoice_redaction.ipynb       # Original Jupyter notebook
├── usage_example.py              # Command-line usage examples
├── requirements.txt              # Python dependencies
├── test*.png                     # Sample invoice images
└── README.md                     # This file
```

## Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **OpenCV ImportError: libGL.so.1 cannot open shared object**
   ```bash
   # Option 1: Install system dependencies
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   
   # Option 2: Use headless OpenCV (recommended)
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **PDF processing fails**
   - Try enabling "Force OCR for PDF" option
   - Ensure poppler-utils is installed

4. **OCR not working**
   - Verify Tesseract installation: `tesseract --version`
   - Check image quality and resolution

5. **Poor detection accuracy**
   - Use higher resolution images
   - Ensure text is clear and not skewed
   - Try different OCR configurations

### Performance Tips

- For large PDFs, processing may take several minutes
- Lower resolution images process faster but with reduced accuracy
- Enable OCR for image-based PDFs for better results

## Limitations

- Optimized for US address and phone number formats
- OCR accuracy depends on image quality
- May occasionally flag business data as PII (false positives)
- Processing time increases with file size and complexity

## Security Note

This tool is designed for defensive security purposes - protecting sensitive information by redacting it from documents. All processing is done locally and no data is transmitted externally.