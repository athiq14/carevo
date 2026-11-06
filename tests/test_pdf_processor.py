"""
Comprehensive unit tests for PDFProcessor and ResumeExtractionResult.
Programmatically generates test PDFs using PyMuPDF without real candidate data.
"""
import io
import pytest
import pymupdf

from models.resume import ExtractionStatus, ResumeExtractionResult
from processors.pdf_processor import (
    CorruptedPDFError,
    EmptyFileError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFProcessor,
)


def make_pdf_bytes(pages: list[str]) -> bytes:
    """Helper to programmatically generate an in-memory PDF with given page texts."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_encrypted_pdf_bytes(password: str = "testpass123") -> bytes:
    """Helper to programmatically generate an encrypted in-memory PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Confidential Resume Data")
    enc_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw="owner123",
    )
    doc.close()
    return enc_bytes


@pytest.fixture
def processor() -> PDFProcessor:
    return PDFProcessor(min_total_chars=80, min_chars_per_page=40)


def test_valid_text_based_pdf(processor: PDFProcessor):
    """Verify standard text extraction from a valid resume PDF."""
    sample_text = (
        "Senior Cloud Architect\n"
        "Expertise in distributed systems, Kubernetes, CI/CD pipelines, and Python backend services.\n"
        "Over 10 years of experience designing scalable microservice architectures."
    )
    pdf_bytes = make_pdf_bytes([sample_text])

    result = processor.extract_text(pdf_bytes, filename="resume_valid.pdf")

    assert isinstance(result, ResumeExtractionResult)
    assert result.filename == "resume_valid.pdf"
    assert result.page_count == 1
    assert "Senior Cloud Architect" in result.extracted_text
    assert result.character_count == len(result.extracted_text)
    assert result.requires_ocr is False
    assert result.extraction_status == ExtractionStatus.SUCCESS
    assert result.error_message is None


def test_stream_input(processor: PDFProcessor):
    """Verify that file-like streams (io.BytesIO or UploadedFile) are supported."""
    sample_text = (
        "Principal Systems Engineer with extensive leadership and architecture experience.\n"
        "Skilled in Go, Python, Linux kernel performance tuning, and cloud computing."
    )
    stream = io.BytesIO(make_pdf_bytes([sample_text]))

    result = processor.extract_text(stream, filename="streamed_resume.pdf")

    assert result.filename == "streamed_resume.pdf"
    assert result.page_count == 1
    assert "Principal Systems Engineer" in result.extracted_text
    # Verify stream pointer can be re-read
    assert len(stream.read()) == len(stream.getvalue())


def test_empty_file_rejected(processor: PDFProcessor):
    """Verify that an empty 0-byte file raises EmptyFileError."""
    with pytest.raises(EmptyFileError, match="empty"):
        processor.extract_text(b"", filename="empty.pdf")


def test_invalid_pdf_bytes_rejected(processor: PDFProcessor):
    """Verify that non-PDF content raises InvalidPDFError."""
    garbage_bytes = b"This is plain text and definitely not a valid PDF file format."
    with pytest.raises(InvalidPDFError, match="header missing"):
        processor.extract_text(garbage_bytes, filename="bad.pdf")


def test_invalid_extension_rejected(processor: PDFProcessor):
    """Verify that non-.pdf extensions are rejected."""
    sample_text = "Some valid resume text content."
    pdf_bytes = make_pdf_bytes([sample_text])
    with pytest.raises(InvalidPDFError, match="Only .pdf files are supported"):
        processor.extract_text(pdf_bytes, filename="resume.docx")


def test_encrypted_pdf_rejected(processor: PDFProcessor):
    """Verify that password-protected PDFs are detected and raise EncryptedPDFError."""
    enc_bytes = make_encrypted_pdf_bytes("secure_password")
    with pytest.raises(EncryptedPDFError, match="password-protected or encrypted"):
        processor.extract_text(enc_bytes, filename="locked_resume.pdf")


def test_multi_page_pdf(processor: PDFProcessor):
    """Verify correct page count and concatenated text across multiple pages."""
    p1 = "Page One: Engineering Leadership, team management, and strategy.\nExtensive career overview."
    p2 = "Page Two: Work History and enterprise project delivery at major tech organizations."
    p3 = "Page Three: Education, certifications, and technical patents."
    pdf_bytes = make_pdf_bytes([p1, p2, p3])

    result = processor.extract_text(pdf_bytes, filename="multipage_resume.pdf")

    assert result.page_count == 3
    assert "Page One" in result.extracted_text
    assert "Page Two" in result.extracted_text
    assert "Page Three" in result.extracted_text
    assert result.requires_ocr is False
    assert result.extraction_status == ExtractionStatus.SUCCESS


def test_insufficient_text_detects_ocr_need(processor: PDFProcessor):
    """Verify that PDFs with sparse or image-only content flag requires_ocr."""
    # Only 5 characters on 2 pages -> well below the threshold
    pdf_bytes = make_pdf_bytes(["Hi", ""])

    result = processor.extract_text(pdf_bytes, filename="scanned_resume.pdf")

    assert result.page_count == 2
    assert result.requires_ocr is True
    assert result.extraction_status == ExtractionStatus.WARNING
    assert result.error_message is not None
    assert "scanned image" in result.error_message


def test_character_and_page_count_accuracy(processor: PDFProcessor):
    """Verify exact character and page count calculations."""
    text1 = "Line A: Systems architect and designer."
    text2 = "Line B: Data pipeline optimization specialist."
    pdf_bytes = make_pdf_bytes([text1, text2])

    result = processor.extract_text(pdf_bytes, filename="counts_test.pdf")

    assert result.page_count == 2
    assert result.character_count == len(result.extracted_text)
    assert result.character_count > 0

