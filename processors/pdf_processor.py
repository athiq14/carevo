"""
PDF processing and validation service for resume extraction.
Provides in-memory parsing, structural checks, and OCR readiness detection using PyMuPDF.
"""
from typing import BinaryIO, Union
import pymupdf

from models.resume import ExtractionStatus, ResumeExtractionResult


class PDFProcessingError(Exception):
    """Base exception for all PDF processing failures."""
    pass


class EmptyFileError(PDFProcessingError):
    """Raised when the provided PDF file contains 0 bytes."""
    pass


class InvalidPDFError(PDFProcessingError):
    """Raised when the file format, extension, or header signature is not a valid PDF."""
    pass


class EncryptedPDFError(PDFProcessingError):
    """Raised when the PDF document is password-protected or encrypted."""
    pass


class CorruptedPDFError(PDFProcessingError):
    """Raised when the PDF structure is malformed or unreadable."""
    pass


class PDFProcessor:
    """
    Reusable service for validating and extracting text and metadata from PDF resumes.
    """

    def __init__(
        self,
        min_total_chars: int = 80,
        min_chars_per_page: int = 40,
    ) -> None:
        """
        Initialize the PDF processor with OCR detection heuristics.

        Args:
            min_total_chars: Minimum total non-whitespace characters expected.
            min_chars_per_page: Minimum non-whitespace characters per page expected.
        """
        self.min_total_chars = min_total_chars
        self.min_chars_per_page = min_chars_per_page

    def extract_text(
        self,
        file_input: Union[bytes, BinaryIO],
        filename: str = "resume.pdf",
    ) -> ResumeExtractionResult:
        """
        Validate, parse, and extract text and structural metadata from a PDF.

        Args:
            file_input: Raw PDF bytes or a readable binary stream.
            filename: Name of the uploaded file for logging and metadata.

        Returns:
            ResumeExtractionResult: Structured extraction result with metadata and OCR indicator.

        Raises:
            EmptyFileError: If file is empty.
            InvalidPDFError: If file lacks PDF extension or magic bytes.
            EncryptedPDFError: If file is password protected.
            CorruptedPDFError: If PDF cannot be opened or parsed.
        """
        # 1. Validate file extension if filename is provided
        clean_filename = filename.strip() if filename else "resume.pdf"
        if not clean_filename.lower().endswith(".pdf"):
            raise InvalidPDFError(
                f"Invalid file extension for '{clean_filename}'. Only .pdf files are supported."
            )

        # 2. Ingest binary content into memory
        content: bytes
        if isinstance(file_input, bytes):
            content = file_input
        elif hasattr(file_input, "read"):
            current_pos = None
            if hasattr(file_input, "tell") and hasattr(file_input, "seek"):
                try:
                    current_pos = file_input.tell()
                except Exception:
                    current_pos = None

            content = file_input.read()

            # Restore stream position if seekable
            if current_pos is not None and hasattr(file_input, "seek"):
                try:
                    file_input.seek(current_pos)
                except Exception:
                    pass
        else:
            raise InvalidPDFError("Input must be either raw bytes or a readable binary stream.")

        # 3. Check for empty payload
        if len(content) == 0:
            raise EmptyFileError("The uploaded PDF file is empty (0 bytes).")

        # 4. Check for PDF magic bytes signature (%PDF-)
        if b"%PDF-" not in content[:1024]:
            raise InvalidPDFError(
                "The file signature does not match a valid PDF document (%PDF- header missing)."
            )

        # 5. Open document safely with PyMuPDF
        doc: pymupdf.Document
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise CorruptedPDFError(
                f"Failed to open PDF document: {exc}"
            ) from exc

        try:
            # 6. Check for encryption or password protection
            if doc.is_encrypted or doc.needs_pass:
                raise EncryptedPDFError(
                    "The PDF document is password-protected or encrypted and cannot be parsed."
                )

            # 7. Check page count
            page_count = doc.page_count
            if page_count == 0:
                raise CorruptedPDFError("The PDF document contains 0 readable pages.")

            # 8. Extract text page by page
            page_texts: list[str] = []
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                page_text = page.get_text("text") or ""
                page_texts.append(page_text.strip())

            extracted_text = "\n\n".join(t for t in page_texts if t).strip()
            character_count = len(extracted_text)

            # Count non-whitespace characters for OCR detection
            non_ws_char_count = len("".join(extracted_text.split()))

            # 9. Evaluate OCR requirement (scanned or image-only documents)
            requires_ocr = (
                non_ws_char_count < self.min_total_chars
                or (page_count > 0 and (non_ws_char_count / page_count) < self.min_chars_per_page)
            )

            status = ExtractionStatus.WARNING if requires_ocr else ExtractionStatus.SUCCESS
            diagnostic_message = None
            if requires_ocr:
                diagnostic_message = (
                    "Document appears to be a scanned image or contains insufficient text. "
                    "OCR may be required for complete career intelligence analysis."
                )

            return ResumeExtractionResult(
                filename=clean_filename,
                page_count=page_count,
                extracted_text=extracted_text,
                character_count=character_count,
                extraction_status=status,
                requires_ocr=requires_ocr,
                error_message=diagnostic_message,
            )

        finally:
            # 10. Guarantee document handle closure
            if doc is not None:
                doc.close()

