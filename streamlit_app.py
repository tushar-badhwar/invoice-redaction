import streamlit as st
import tempfile
import os
import io

# Set environment variables early
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

st.set_page_config(
    page_title="Invoice PII Redaction Tool",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy loading functions
@st.cache_resource
def load_dependencies():
    """Load heavy dependencies only when needed"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        from PIL import Image
        import numpy as np
        import cv2
        
        return True, "All dependencies loaded successfully"
    except Exception as e:
        return False, f"Dependency error: {e}"

@st.cache_resource  
def load_redaction_functions():
    """Load redaction functions only when needed"""
    try:
        from invoice_redaction_clean import process_invoice, detect_file_type, process_pdf_invoice
        return process_invoice, detect_file_type, process_pdf_invoice, None
    except Exception as e:
        return None, None, None, str(e)

def process_file(file_path, file_name, file_size, use_ocr_for_pdf):
    """Process a file (either uploaded or demo) and display results"""
    
    # Load dependencies first
    with st.spinner("Loading dependencies..."):
        deps_loaded, deps_msg = load_dependencies()
        if not deps_loaded:
            st.error(f"❌ {deps_msg}")
            return
        
        process_invoice, detect_file_type, process_pdf_invoice, func_error = load_redaction_functions()
        if func_error:
            st.error(f"❌ Failed to load redaction functions: {func_error}")
            return
    
    try:
        # Display original file info
        st.subheader("📄 File Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", file_name)
        with col2:
            st.metric("File Size", f"{file_size / 1024:.1f} KB")
        with col3:
            file_type = detect_file_type(file_path)
            st.metric("File Type", file_type.upper())
        
        # Process the file
        st.subheader("🔄 Processing")
        
        with st.spinner("Processing invoice... This may take a moment."):
            if file_type == 'pdf':
                redacted_pages, stats = process_pdf_invoice(file_path, use_ocr_for_pdf)
                
                if redacted_pages:
                    st.success("✅ PDF processing completed successfully!")
                    st.info("PDF results display simplified for demo")
                else:
                    st.error("❌ Failed to process the PDF. Please try with OCR enabled.")
                    
            else:
                redacted_image, stats = process_invoice(file_path)
                
                if redacted_image is not None:
                    st.success("✅ Image processing completed successfully!")
                    st.info("Image results display simplified for demo")
                else:
                    st.error("❌ Failed to process the image. Please check the file format and quality.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

def main():
    st.title("🔒 Invoice PII Redaction Tool")
    st.markdown("**Automatically detect and redact personally identifiable information (PII) from invoices**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")
    
    # Demo section
    st.sidebar.subheader("🎯 Try a Demo")
    
    # Available sample files
    sample_files = ['test1.png', 'test2.png', 'test3.png']
    
    if sample_files:
        demo_option = st.sidebar.selectbox(
            "Choose a sample invoice:",
            ["None"] + sample_files,
            help="Select a sample invoice to see how the redaction works"
        )
        
        if demo_option != "None":
            st.sidebar.success(f"Selected: {demo_option}")
            process_demo_file = st.sidebar.button("🚀 Process Sample Invoice", type="primary")
        else:
            process_demo_file = False
            demo_option = None
    else:
        demo_option = None
        process_demo_file = False
    
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your own invoice file",
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
    
    # Process demo file or uploaded file
    if process_demo_file and demo_option:
        process_file(demo_option, demo_option, os.path.getsize(demo_option), use_ocr_for_pdf)
    elif uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            process_file(temp_path, uploaded_file.name, uploaded_file.size, use_ocr_for_pdf)
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
        
        # Highlight demo option if available
        if sample_files:
            st.info("""
            🎯 **Try the Demo!** Select a sample invoice from the sidebar to see how the redaction works instantly.
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How to use:
            1. **Try Demo** - Select a sample invoice from sidebar, or
            2. **Upload** your own invoice file (image or PDF)
            3. **Configure** processing options if needed
            4. **Review** the detected PII and redacted result
            5. **Download** the redacted version
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
        
        # Show sample images preview if available
        if sample_files:
            st.subheader("📋 Sample Invoices Available")
            st.markdown("Preview of available sample invoices for testing:")
            
            # Show preview of first few samples
            preview_cols = st.columns(min(3, len(sample_files)))
            for i, sample_file in enumerate(sample_files[:3]):
                with preview_cols[i]:
                    try:
                        from PIL import Image
                        img = Image.open(sample_file)
                        st.image(img, caption=sample_file, use_column_width=True)
                    except:
                        st.text(f"📄 {sample_file}")
            
            if len(sample_files) > 3:
                st.markdown(f"*...and {len(sample_files) - 3} more samples available in the sidebar*")
        
        st.markdown("---")
        st.info("💡 **Tip:** For best results with PDFs, try enabling 'Force OCR for PDF' if the automatic processing doesn't work well.")

if __name__ == "__main__":
    main()