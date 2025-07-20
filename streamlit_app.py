import streamlit as st
import tempfile
import os
import io
from PIL import Image
import numpy as np

# Fix for OpenCV and matplotlib headless import issues
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

try:
    import cv2
except ImportError as e:
    st.error(f"""
    OpenCV import error: {e}
    
    Please install the required system dependencies:
    ```bash
    sudo apt-get update
    sudo apt-get install libgl1-mesa-glx libglib2.0-0
    ```
    
    Or try installing opencv-python-headless instead:
    ```bash
    pip uninstall opencv-python
    pip install opencv-python-headless
    ```
    """)
    st.stop()

# Import the invoice redaction functions
from invoice_redaction_clean import (
    process_invoice_universal,
    process_invoice,
    detect_file_type,
    process_pdf_invoice,
    extract_text_with_boxes,
    detect_names,
    detect_addresses,
    detect_phone_numbers,
    detect_bank_accounts,
    detect_emails,
    redact_image
)

st.set_page_config(
    page_title="Invoice PII Redaction Tool",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔒 Invoice PII Redaction Tool")
    st.markdown("**Automatically detect and redact personally identifiable information (PII) from invoices**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an invoice file",
        type=['png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, PDF, TIFF, BMP"
    )
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    
    if uploaded_file and uploaded_file.name.lower().endswith('.pdf'):
        use_ocr_for_pdf = st.sidebar.checkbox(
            "Force OCR for PDF",
            value=False,
            help="Use OCR instead of native text extraction for PDFs. Recommended for image-based PDFs."
        )
    else:
        use_ocr_for_pdf = False
    
    # Information section
    with st.sidebar.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool automatically detects and redacts:
        - **Names** (using NER)
        - **Addresses** (US format)
        - **Phone numbers** (US format)
        - **Email addresses**
        - **Bank account numbers**
        
        **Features:**
        - Preserves business data and amounts
        - Handles both images and PDFs
        - Side-by-side comparison
        - Downloadable results
        """)
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Display original file info
            st.subheader("📄 File Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                file_type = detect_file_type(temp_path)
                st.metric("File Type", file_type.upper())
            
            # Process the file
            st.subheader("🔄 Processing")
            
            with st.spinner("Processing invoice... This may take a moment."):
                if file_type == 'pdf':
                    redacted_pages, stats = process_pdf_invoice(temp_path, use_ocr_for_pdf)
                    
                    if redacted_pages:
                        st.success("✅ PDF processing completed successfully!")
                        display_pdf_results_streamlit(temp_path, redacted_pages, stats)
                    else:
                        st.error("❌ Failed to process the PDF. Please try with OCR enabled.")
                        
                else:
                    redacted_image, stats = process_invoice(temp_path)
                    
                    if redacted_image is not None:
                        st.success("✅ Image processing completed successfully!")
                        display_image_results_streamlit(temp_path, redacted_image, stats)
                    else:
                        st.error("❌ Failed to process the image. Please check the file format and quality.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    else:
        # Landing page content
        st.markdown("---")
        st.subheader("🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How to use:
            1. **Upload** an invoice file (image or PDF)
            2. **Configure** processing options if needed
            3. **Review** the detected PII and redacted result
            4. **Download** the redacted version
            """)
        
        with col2:
            st.markdown("""
            ### What gets redacted:
            - 👤 **Personal names**
            - 🏠 **Addresses**
            - 📞 **Phone numbers**
            - 📧 **Email addresses**
            - 🏦 **Bank account numbers**
            """)
        
        st.markdown("---")
        st.info("💡 **Tip:** For best results with PDFs, try enabling 'Force OCR for PDF' if the automatic processing doesn't work well.")

def display_image_results_streamlit(original_path, redacted_image, stats):
    """Display image processing results in Streamlit"""
    
    # Statistics
    display_statistics(stats)
    
    st.subheader("📊 Results Comparison")
    
    # Load original image
    original = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Invoice**")
        st.image(original_rgb, use_column_width=True)
    
    with col2:
        st.markdown("**Redacted Invoice**")
        st.image(redacted_image, use_column_width=True)
    
    # Download button
    provide_download_button(redacted_image, "redacted_invoice.png")

def display_pdf_results_streamlit(pdf_path, redacted_pages, stats):
    """Display PDF processing results in Streamlit"""
    
    # Statistics
    display_statistics(stats)
    
    st.subheader("📊 Results Comparison")
    
    # Import here to avoid issues if pdf2image not available
    import pdf2image
    
    # Load original pages
    original_images = pdf2image.convert_from_path(pdf_path, dpi=150)  # Lower DPI for web display
    
    # Display each page
    for page_num, (original, redacted) in enumerate(zip(original_images, redacted_pages), 1):
        st.markdown(f"**Page {page_num}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Original*")
            st.image(original, use_column_width=True)
        
        with col2:
            st.markdown("*Redacted*")
            st.image(redacted, use_column_width=True)
        
        if page_num < len(original_images):
            st.markdown("---")
    
    # Provide download for each page
    for i, redacted_page in enumerate(redacted_pages, 1):
        provide_download_button(redacted_page, f"redacted_page_{i}.png", key=f"page_{i}")

def display_statistics(stats):
    """Display redaction statistics"""
    st.subheader("📈 Redaction Statistics")
    
    if stats.get('total_redacted', 0) > 0:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("👤 Names", stats.get('names', 0))
        with col2:
            st.metric("🏠 Addresses", stats.get('addresses', 0))
        with col3:
            st.metric("📞 Phones", stats.get('phones', 0))
        with col4:
            st.metric("📧 Emails", stats.get('emails', 0))
        with col5:
            st.metric("🏦 Bank Accounts", stats.get('bank_accounts', 0))
        with col6:
            st.metric("🔒 Total Redacted", stats.get('total_redacted', 0))
        
        # Create a simple breakdown chart
        if any(stats.get(key, 0) > 0 for key in ['names', 'addresses', 'phones', 'emails', 'bank_accounts']):
            chart_data = {
                'PII Type': [],
                'Count': []
            }
            
            pii_types = {
                'Names': stats.get('names', 0),
                'Addresses': stats.get('addresses', 0),
                'Phones': stats.get('phones', 0),
                'Emails': stats.get('emails', 0),
                'Bank Accounts': stats.get('bank_accounts', 0)
            }
            
            for pii_type, count in pii_types.items():
                if count > 0:
                    chart_data['PII Type'].append(pii_type)
                    chart_data['Count'].append(count)
            
            if chart_data['PII Type']:
                st.bar_chart(data=chart_data, x='PII Type', y='Count')
    else:
        st.info("ℹ️ No PII detected in this document.")

def provide_download_button(image_array, filename, key=None):
    """Provide download button for redacted image"""
    
    # Convert numpy array to PIL Image
    if isinstance(image_array, np.ndarray):
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        pil_image = Image.fromarray(image_array)
    else:
        pil_image = image_array
    
    # Convert to bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    # Download button
    st.download_button(
        label=f"⬇️ Download {filename}",
        data=img_bytes,
        file_name=filename,
        mime="image/png",
        key=key
    )

if __name__ == "__main__":
    main()