"""
Deterministic resume section detection engine.
Identifies canonical resume sections and unknown headings using normalized pattern matching.
"""
import re
from typing import Optional

from models.resume_sections import (
    ParsedResumeSections,
    ResumeSection,
    SectionType,
)
from processors.text_cleaner import clean_resume_text


# Canonical list of resume sections excluding UNKNOWN
CANONICAL_SECTIONS: list[SectionType] = [
    SectionType.CONTACT_INFORMATION,
    SectionType.PROFESSIONAL_SUMMARY,
    SectionType.CAREER_OBJECTIVE,
    SectionType.TECHNICAL_SKILLS,
    SectionType.WORK_EXPERIENCE,
    SectionType.INTERNSHIPS,
    SectionType.EDUCATION,
    SectionType.PROJECTS,
    SectionType.CERTIFICATIONS,
    SectionType.ACHIEVEMENTS,
    SectionType.PUBLICATIONS,
    SectionType.LANGUAGES,
    SectionType.OTHER_INFORMATION,
]


DEFAULT_SECTION_ALIASES: dict[SectionType, set[str]] = {
    SectionType.CONTACT_INFORMATION: {
        "contact",
        "contact information",
        "contact info",
        "contact details",
        "contacts",
        "personal details",
        "personal information",
        "personal info",
        "get in touch",
    },
    SectionType.PROFESSIONAL_SUMMARY: {
        "professional summary",
        "summary",
        "summary of qualifications",
        "professional profile",
        "profile",
        "career summary",
        "executive summary",
        "about me",
        "personal statement",
        "biography",
        "overview",
        "qualifications summary",
    },
    SectionType.CAREER_OBJECTIVE: {
        "career objective",
        "objective",
        "professional objective",
        "job objective",
    },
    SectionType.TECHNICAL_SKILLS: {
        "technical skills",
        "skills",
        "skills & competencies",
        "skills and competencies",
        "core competencies",
        "competencies",
        "key skills",
        "areas of expertise",
        "technical proficiencies",
        "technical expertise",
        "technologies",
        "tech stack",
        "tools & technologies",
        "tools and technologies",
        "programming skills",
        "proficiencies",
    },
    SectionType.WORK_EXPERIENCE: {
        "work experience",
        "experience",
        "professional experience",
        "employment history",
        "work history",
        "career history",
        "professional background",
        "relevant experience",
        "employment",
        "work background",
    },
    SectionType.INTERNSHIPS: {
        "internships",
        "internship experience",
        "training & internships",
        "internship",
        "internship history",
        "trainee experience",
    },
    SectionType.EDUCATION: {
        "education",
        "academic background",
        "educational background",
        "academic qualifications",
        "academics",
        "education & credentials",
        "education and credentials",
        "degrees",
        "qualifications",
    },
    SectionType.PROJECTS: {
        "projects",
        "personal projects",
        "key projects",
        "academic projects",
        "technical projects",
        "selected projects",
        "notable projects",
        "major projects",
    },
    SectionType.CERTIFICATIONS: {
        "certifications",
        "licenses & certifications",
        "licenses and certifications",
        "certificates",
        "accreditations",
        "professional certifications",
        "credentials",
        "licensures",
    },
    SectionType.ACHIEVEMENTS: {
        "achievements",
        "honors & awards",
        "honors and awards",
        "awards & honors",
        "awards and honors",
        "awards",
        "key achievements",
        "accomplishments",
        "recognition",
        "honors",
    },
    SectionType.PUBLICATIONS: {
        "publications",
        "research & publications",
        "research and publications",
        "papers",
        "conference proceedings",
        "published works",
        "publications & patents",
        "publications and patents",
        "patents",
        "patents & publications",
        "patents and publications",
    },
    SectionType.LANGUAGES: {
        "languages",
        "language proficiency",
        "languages known",
        "spoken languages",
    },
    SectionType.OTHER_INFORMATION: {
        "other information",
        "additional information",
        "interests",
        "hobbies",
        "activities",
        "volunteer experience",
        "volunteering",
        "volunteer work",
        "leadership",
        "extracurricular activities",
        "affiliations",
        "memberships",
    },
}


class SectionDetector:
    """
    Deterministic rule-based resume section parser.
    Identifies section headers and segments resume text accurately.
    """

    def __init__(
        self,
        custom_aliases: Optional[dict[SectionType, set[str]]] = None,
    ) -> None:
        """
        Initialize the detector with default or extended section aliases.
        """
        self.aliases: dict[SectionType, set[str]] = (
            custom_aliases if custom_aliases is not None else DEFAULT_SECTION_ALIASES
        )
        # Build inverted lookup for O(1) matching
        self._lookup: dict[str, SectionType] = {}
        for section_type, alias_set in self.aliases.items():
            for alias in alias_set:
                self._lookup[alias.lower().strip()] = section_type

    def _is_heading(self, line: str) -> Optional[tuple[SectionType, str]]:
        """
        Determine if a line is a section heading.

        Returns:
            Optional[tuple[SectionType, str]]: (Detected SectionType, original heading string)
        """
        clean_line = line.strip()
        if not clean_line:
            return None

        # Bullet points are list items, never section headings
        if clean_line.startswith(("-", "*", "•", "●", "○", "■", "▪", "▫", "◆", "◇", "►", "▸", "–", "—", "\ufffd", "\uf0b7", "\uf0a7", "\xb7")):
            return None

        # Check line length constraint: headings are concise
        words = clean_line.split()
        if len(words) > 5 or len(clean_line) > 50:
            return None

        # Reject sentences ending in common punctuation
        if clean_line.endswith((".", ",", ";", "?", "!")):
            return None

        # Strip markdown markers, numbering, or leading prefixes: e.g. "## Education", "1. Skills"
        candidate = re.sub(r'^(?:[0-9]+[\.\)]|\#+)\s*', '', clean_line).strip()
        if not candidate:
            return None

        # Strip trailing colon or dashes
        candidate_normalized = candidate.rstrip(":- ").strip().lower()

        # 1. Direct alias lookup against known canonical section names
        if candidate_normalized in self._lookup:
            return self._lookup[candidate_normalized], clean_line

        # 2. Heuristic for unknown headings:
        # Must have clear heading traits (ends in colon, markdown header, or multi-word/lengthy all-caps)
        has_colon = clean_line.endswith(":")
        has_markdown = clean_line.startswith(("#", "##", "###"))
        
        # All-caps line must be either multi-word (e.g. 'PATENT PORTFOLIO') or lengthy (>= 7 chars e.g. 'REFERENCES')
        # to prevent single acronyms ('SQL', 'AWS', 'API') or metric/grade lines ('GPA: 3.92 / 4.0') from being misclassified
        is_all_caps = (
            candidate.isupper()
            and any(c.isalpha() for c in candidate)
            and not any(c.isdigit() for c in candidate)
            and (len(words) >= 2 or len(candidate) >= 7)
        )

        if (has_colon or is_all_caps or has_markdown) and len(words) <= 4:
            return SectionType.UNKNOWN, clean_line

        return None

    def _is_contact_info(self, text: str) -> bool:
        """Helper to identify if unheaded preamble text resembles contact information."""
        has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text))
        has_phone = bool(re.search(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))
        has_url = bool(re.search(r'https?://|www\.|linkedin\.com|github\.com', text, re.IGNORECASE))
        return has_email or has_phone or has_url

    def detect_sections(
        self,
        cleaned_text: str,
        original_text: Optional[str] = None,
    ) -> ParsedResumeSections:
        """
        Segment cleaned resume text into distinct structured sections.

        Args:
            cleaned_text: Cleaned and normalized resume text.
            original_text: Optional raw extracted text from the PDF.

        Returns:
            ParsedResumeSections: Structured representation of all sections.
        """
        orig_text = original_text if original_text is not None else cleaned_text
        if not cleaned_text or not cleaned_text.strip():
            return ParsedResumeSections(
                original_text=orig_text,
                cleaned_text=cleaned_text,
                sections=[],
                detected_section_names=[],
                undetected_section_names=[s.value for s in CANONICAL_SECTIONS],
            )

        lines = cleaned_text.splitlines()

        # Track sections during linear scan
        sections: list[ResumeSection] = []

        current_type: Optional[SectionType] = None
        current_heading: Optional[str] = None
        current_content_lines: list[str] = []
        current_start_line: int = 1
        current_end_line: int = 1

        def commit_section():
            nonlocal current_type, current_heading, current_content_lines, current_start_line, current_end_line
            content_str = "\n".join(current_content_lines).strip()
            if content_str or current_heading:
                sec_type = current_type
                heading_val = current_heading

                # Handle top-of-resume preamble or candidate name heading followed by contact info
                if len(sections) == 0:
                    if sec_type is None and self._is_contact_info(content_str):
                        sec_type = SectionType.CONTACT_INFORMATION
                    elif sec_type == SectionType.UNKNOWN and self._is_contact_info(content_str):
                        sec_type = SectionType.CONTACT_INFORMATION
                        if heading_val:
                            content_str = f"{heading_val}\n{content_str}".strip()
                            heading_val = None

                if sec_type is None:
                    sec_type = SectionType.UNKNOWN

                sections.append(
                    ResumeSection(
                        section_type=sec_type,
                        original_heading=heading_val,
                        content=content_str,
                        start_line=current_start_line,
                        end_line=current_end_line,
                    )
                )

        for line_idx, line in enumerate(lines, start=1):
            heading_match = self._is_heading(line)

            if heading_match is not None:
                detected_type, heading_str = heading_match

                # Commit previous section if any content exists
                if current_type is not None or current_content_lines:
                    commit_section()

                # Start new section
                current_type = detected_type
                current_heading = heading_str
                current_content_lines = []
                current_start_line = line_idx
                current_end_line = line_idx
            else:
                # Regular content line
                if current_type is None and not current_content_lines:
                    current_start_line = line_idx
                current_content_lines.append(line)
                if line.strip():
                    current_end_line = line_idx

        # Commit final trailing section
        if current_type is not None or current_content_lines:
            commit_section()

        # Compute detected vs undetected section names
        detected_names: list[str] = []
        seen = set()
        for sec in sections:
            if sec.section_type != SectionType.UNKNOWN and sec.section_type.value not in seen:
                detected_names.append(sec.section_type.value)
                seen.add(sec.section_type.value)

        undetected_names = [
            s.value for s in CANONICAL_SECTIONS if s.value not in seen
        ]

        return ParsedResumeSections(
            original_text=orig_text,
            cleaned_text=cleaned_text,
            sections=sections,
            detected_section_names=detected_names,
            undetected_section_names=undetected_names,
        )


def detect_resume_sections(
    cleaned_text: str,
    original_text: Optional[str] = None,
) -> ParsedResumeSections:
    """Convenience functional wrapper for SectionDetector.detect_sections."""
    detector = SectionDetector()
    return detector.detect_sections(cleaned_text, original_text)
