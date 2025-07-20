import streamlit as st
import tempfile
import os
import io
from PIL import Image
import numpy as np
import cv2

# Import the invoice redaction functions
from invoice_redaction_clean import (
    process_invoice,
    detect_file_type,
    process_pdf_invoice
)

st.set_page_config(
    page_title="Invoice PII Redaction Tool",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_results(original_path: str, redacted_image: np.ndarray, stats: dict):
    """Display the original and redacted images side by side."""
    try:
        # Load original image
        original = cv2.imread(original_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Display statistics
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
        else:
            st.info("ℹ️ No PII detected in this document.")
        
        # Create side-by-side comparison
        st.subheader("📊 Results Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Invoice**")
            st.image(original_rgb, use_column_width=True)
        
        with col2:
            st.markdown("**Redacted Invoice**")
            st.image(redacted_image, use_column_width=True)
        
        # Download button
        pil_image = Image.fromarray(redacted_image.astype(np.uint8))
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        st.download_button(
            label="⬇️ Download Redacted Invoice",
            data=img_bytes,
            file_name="redacted_invoice.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

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
                        # For now, just show the first page
                        if redacted_pages:
                            st.info("Displaying first page of PDF")
                            display_results(temp_path, redacted_pages[0], stats)
                    else:
                        st.error("❌ Failed to process the PDF. Please try with OCR enabled.")
                        
                else:
                    redacted_image, stats = process_invoice(temp_path)
                    
                    if redacted_image is not None:
                        st.success("✅ Image processing completed successfully!")
                        display_results(temp_path, redacted_image, stats)
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

if __name__ == "__main__":
    main()