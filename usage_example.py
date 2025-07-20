#!/usr/bin/env python3
"""
Example usage of the Invoice Redaction Tool
"""

from invoice_redaction_clean import process_invoice_universal, process_invoice_from_path

def main():
    """
    Example usage of the invoice redaction tool
    """
    print("Invoice Redaction Tool - Usage Examples")
    print("=" * 50)
    
    # Example 1: Process an image file
    print("\n1. Processing image file:")
    process_invoice_from_path('invoice-sample-1.png')
    
    # Example 2: Process a PDF file (auto-detect method)
    print("\n2. Processing PDF file (auto-detect):")
    # process_invoice_universal('invoice.pdf')
    
    # Example 3: Process a PDF file with OCR
    print("\n3. Processing PDF file with OCR:")
    # process_invoice_universal('invoice.pdf', use_ocr_for_pdf=True)
    
    # Example 4: Process multiple files
    print("\n4. Processing multiple files:")
    image_files = ['invoice-sample-1.png', 'invoice-sample-2.png', 'invoice-sample-3.png', 'invoice-sample-4.png']
    
    for file_path in image_files:
        print(f"\nProcessing {file_path}:")
        try:
            process_invoice_universal(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()