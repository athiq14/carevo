"""
Resume processing pipeline orchestrating PDF extraction, text cleaning,
section detection, and entity extraction into a unified ResumeProfile.
"""
from typing import BinaryIO, Optional, Union

from models.resume import ExtractionStatus, ResumeExtractionResult
from models.resume_sections import ParsedResumeSections
from models.resume_entities import ResumeEntities
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from processors.pdf_processor import PDFProcessingError, PDFProcessor
from processors.text_cleaner import ResumeTextCleaner, clean_resume_text
from processors.section_detector import CANONICAL_SECTIONS, SectionDetector, detect_resume_sections
from processors.entity_extractor import EntityExtractor, extract_resume_entities


class ResumePipeline:
    """
    End-to-end pipeline coordinating document extraction, sanitization,
    structural parsing, and deterministic entity extraction into an immutable ResumeProfile.
    """

    def __init__(
        self,
        pdf_processor: Optional[PDFProcessor] = None,
        text_cleaner: Optional[ResumeTextCleaner] = None,
        section_detector: Optional[SectionDetector] = None,
        entity_extractor: Optional[EntityExtractor] = None,
    ) -> None:
        """
        Initialize pipeline with processor components.
        Allows dependency injection for testing and modularity.
        """
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.text_cleaner = text_cleaner or ResumeTextCleaner()
        self.section_detector = section_detector or SectionDetector()
        self.entity_extractor = entity_extractor or EntityExtractor()

    def process_resume(
        self,
        file_input: Union[bytes, BinaryIO],
        filename: str = "resume.pdf",
        raise_on_error: bool = False,
    ) -> ResumeProfile:
        """
        Execute the end-to-end processing pipeline on an uploaded resume PDF.

        Args:
            file_input: Raw PDF bytes or readable binary stream.
            filename: Name of the uploaded resume document.
            raise_on_error: If True, re-raises PDFProcessingError instead of returning a FAILED profile.

        Returns:
            ResumeProfile: Consolidated, validated profile.
        """
        # Step 1: PDF Extraction and In-Memory Validation
        try:
            pdf_result = self.pdf_processor.extract_text(file_input, filename=filename)
        except PDFProcessingError as err:
            if raise_on_error:
                raise
            return self._build_failed_profile(filename=filename, error_message=str(err))
        except Exception as err:
            if raise_on_error:
                raise
            return self._build_failed_profile(
                filename=filename,
                error_message=f"Unexpected extraction failure: {err}",
            )

        if pdf_result.extraction_status == ExtractionStatus.FAILED:
            return self._build_failed_profile(
                filename=filename,
                error_message=pdf_result.error_message or "Document extraction marked as failed.",
            )

        # Step 2: Deterministic Text Sanitization and Cleaning
        cleaned_text = self.text_cleaner.clean(pdf_result.extracted_text)

        # Step 3: Structural Section Boundary and Heading Detection
        parsed_sections = self.section_detector.detect_sections(
            cleaned_text=cleaned_text,
            original_text=pdf_result.extracted_text,
        )

        # Step 4: Deterministic Entity Extraction
        entities = self.entity_extractor.extract_entities(parsed_sections)

        # Step 5: Lifecycle Status & Extraction Warnings
        warnings: list[str] = []
        ocr_warning: Optional[str] = None

        if pdf_result.requires_ocr:
            ocr_msg = (
                pdf_result.error_message
                or "Document appears to be a scanned image or lacks sufficient extractable text. OCR is recommended."
            )
            warnings.append(ocr_msg)
            ocr_warning = ocr_msg
            status = ProfileProcessingStatus.PARTIAL
        elif not parsed_sections.detected_section_names and len(cleaned_text.strip()) > 0:
            warnings.append("No standard canonical resume sections could be identified.")
            status = ProfileProcessingStatus.PARTIAL
        else:
            status = ProfileProcessingStatus.SUCCESS

        # Check for ambiguity across extracted entities and add notification
        has_ambiguous_skills = any(s.is_ambiguous for s in entities.skills)
        has_ambiguous_exp = any(e.is_ambiguous for e in entities.experience)
        has_ambiguous_edu = any(ed.is_ambiguous for ed in entities.education)
        has_ambiguous_proj = any(p.is_ambiguous for p in entities.projects)

        if has_ambiguous_skills or has_ambiguous_exp or has_ambiguous_edu or has_ambiguous_proj:
            warnings.append("One or more extracted entities contain ambiguous or incomplete information.")

        # Step 6: Assemble Unified Resume Profile
        return ResumeProfile(
            filename=filename,
            page_count=pdf_result.page_count,
            raw_character_count=pdf_result.character_count,
            cleaned_character_count=len(cleaned_text),
            requires_ocr=pdf_result.requires_ocr,
            ocr_warning=ocr_warning,
            status=status,
            warnings=warnings,
            original_text=pdf_result.extracted_text,
            cleaned_text=cleaned_text,
            sections=parsed_sections.sections,
            detected_section_names=parsed_sections.detected_section_names,
            undetected_section_names=parsed_sections.undetected_section_names,
            skills=entities.skills,
            experience=entities.experience,
            education=entities.education,
            projects=entities.projects,
        )

    def _build_failed_profile(self, filename: str, error_message: str) -> ResumeProfile:
        """Construct a consistent FAILED profile without fabricated entities."""
        return ResumeProfile(
            filename=filename,
            page_count=0,
            raw_character_count=0,
            cleaned_character_count=0,
            requires_ocr=False,
            ocr_warning=None,
            status=ProfileProcessingStatus.FAILED,
            warnings=[error_message],
            original_text="",
            cleaned_text="",
            sections=[],
            detected_section_names=[],
            undetected_section_names=[s.value for s in CANONICAL_SECTIONS],
            skills=[],
            experience=[],
            education=[],
            projects=[],
        )


def process_resume_pipeline(
    file_input: Union[bytes, BinaryIO],
    filename: str = "resume.pdf",
    raise_on_error: bool = False,
) -> ResumeProfile:
    """
    Convenience function to execute the full resume pipeline.

    Args:
        file_input: Raw PDF bytes or readable binary stream.
        filename: Name of the uploaded resume document.
        raise_on_error: If True, re-raises extraction errors instead of returning a FAILED profile.

    Returns:
        ResumeProfile: Unified resume profile.
    """
    return ResumePipeline().process_resume(
        file_input=file_input,
        filename=filename,
        raise_on_error=raise_on_error,
    )
