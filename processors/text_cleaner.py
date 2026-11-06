"""
Resume text cleaning and normalization service.
Cleans raw extracted PDF text deterministically without altering factual information.
"""
import re


class ResumeTextCleaner:
    """
    Service for normalizing line endings, whitespace, bullet points, and control characters
    while preserving contact data, punctuation, and structural line boundaries.
    """

    # Characters representing bullet points in PDF documents
    BULLET_CHARS_REGEX = r'[•●○■▪▫◆◇►▸–—\*\ufffd\uf0b7\uf0a7\xb7]'

    # Regex targeting bullet symbols at the beginning of a line (with optional indentation)
    _BULLET_PATTERN = re.compile(rf'^([ \t]*){BULLET_CHARS_REGEX}(?:[ \t]+|\s*)')

    # Unwanted ASCII control characters (excluding \t (9) and \n (10))
    _CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    # Zero-width spaces and formatting markers
    _ZERO_WIDTH_PATTERN = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]')

    # Non-standard spaces (non-breaking spaces, em/en spaces)
    _UNICODE_SPACES_PATTERN = re.compile(r'[\xa0\u2000-\u200a\u3000]')

    # 3 or more consecutive newlines
    _EXCESSIVE_NEWLINES_PATTERN = re.compile(r'\n{3,}')

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Normalize and clean extracted resume text.

        Args:
            text: Raw extracted text from PDF.

        Returns:
            str: Cleaned, normalized text with preserved structural integrity.
        """
        if not text or not text.strip():
            return ""

        # 1. Normalize line endings (CRLF, CR -> LF)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Strip zero-width markers and normalize unicode spaces
        normalized = cls._ZERO_WIDTH_PATTERN.sub("", normalized)
        normalized = cls._UNICODE_SPACES_PATTERN.sub(" ", normalized)

        # 3. Strip unwanted control characters
        normalized = cls._CONTROL_CHARS_PATTERN.sub("", normalized)

        # 4. Process line-by-line for bullets and whitespace normalization
        lines = normalized.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            # Check for bullet point at line start
            bullet_match = cls._BULLET_PATTERN.match(line)
            if bullet_match:
                indent = bullet_match.group(1)
                content = line[bullet_match.end():]
                # Collapse excessive internal spaces
                content = re.sub(r'[ \t]+', ' ', content).strip()
                if content:
                    cleaned_lines.append(f"{indent}- {content}")
                else:
                    cleaned_lines.append("")
            else:
                # Regular text line: collapse internal horizontal whitespace and strip ends
                collapsed = re.sub(r'[ \t]+', ' ', line).strip()
                cleaned_lines.append(collapsed)

        # 5. Join lines and normalize excessive blank lines
        result = "\n".join(cleaned_lines)
        result = cls._EXCESSIVE_NEWLINES_PATTERN.sub("\n\n", result)

        return result.strip()


def clean_resume_text(text: str) -> str:
    """Convenience functional wrapper for ResumeTextCleaner.clean."""
    return ResumeTextCleaner.clean(text)
