"""
Deterministic and semantic Job Description processor.
Extracts, normalizes, deduplicates, and counts job requirements from raw JD text.
Enforces strict input validation, taxonomy canonicalization, and provisional threshold counting.
Supports both deterministic heuristic extraction and StructuredOutputEngine semantic parsing.
"""
from enum import Enum
import re
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from core.llm.exceptions import redact_secrets
from core.llm.models import LLMRequest
from core.llm.structured.engine import StructuredOutputEngine
from models.matching import (
    EvaluationStatus,
    JobDescriptionProfile,
    JobRequirement,
    RequirementCategory,
    RequirementPriority,
)
from processors.entity_extractor import SKILL_TAXONOMY


# =====================================================================
# Canonical Synonym & Taxonomy Lookup
# =====================================================================

def _build_synonym_lookup() -> dict[str, str]:
    """
    Build a flat lookup dictionary mapping lowercase aliases to canonical skill names.
    Derived directly from SKILL_TAXONOMY in processors/entity_extractor.py.
    """
    lookup: dict[str, str] = {}
    for canonical_name, (_, aliases) in SKILL_TAXONOMY.items():
        lookup[canonical_name.lower()] = canonical_name
        for alias in aliases:
            lookup[alias.lower()] = canonical_name
    return lookup


SYNONYM_LOOKUP: dict[str, str] = _build_synonym_lookup()


# Maximum allowed characters for input job descriptions (inputs exceeding this are truncated with a warning)
MAX_JD_CHAR_LIMIT: int = 30_000

# Minimum assessable thresholds
MIN_TECHNICAL_SKILLS_THRESHOLD: int = 3
MIN_TOTAL_REQUIREMENTS_THRESHOLD: int = 4


# =====================================================================
# Intermediate Extraction Schema for LLM Extraction
# =====================================================================

class _RawJDExtraction(BaseModel):
    """Internal Pydantic schema for structured semantic extraction via LLM."""
    model_config = ConfigDict(frozen=True)

    job_title: Optional[str] = Field(default=None, description="Job title stated in the JD.")
    company: Optional[str] = Field(default=None, description="Hiring company stated in the JD.")
    experience_requirement: Optional[str] = Field(default=None, description="Minimum years or tenure required.")
    education_requirement: Optional[str] = Field(default=None, description="Degree or major qualifications.")
    required_skills: list[str] = Field(default_factory=list, description="Must-have mandatory technical skills.")
    preferred_skills: list[str] = Field(default_factory=list, description="Nice-to-have or preferred technical skills.")
    responsibilities: list[str] = Field(default_factory=list, description="Key duties and day-to-day responsibilities.")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications requested.")
    project_requirements: list[str] = Field(default_factory=list, description="Explicit project, portfolio, or GitHub requirements.")


# =====================================================================
# Job Description Processor
# =====================================================================

class JobDescriptionProcessor:
    """
    Processor responsible for parsing, normalizing, deduplicating, and assessing
    qualification requirements from raw job descriptions.
    """

    def __init__(
        self,
        structured_engine: Optional[StructuredOutputEngine] = None,
        enable_llm_extraction: bool = True,
    ) -> None:
        """
        Initialize the JD processor.

        Args:
            structured_engine: Optional StructuredOutputEngine instance for semantic extraction.
            enable_llm_extraction: Whether to attempt LLM extraction when an engine is available.
        """
        self.structured_engine = structured_engine
        self.enable_llm_extraction = enable_llm_extraction

    def validate_input(self, text: str) -> str:
        """
        Validate raw job description input text.

        Raises:
            ValueError: If input is not a string, or is empty/whitespace-only.

        Returns:
            Cleaned string with leading and trailing whitespace stripped.
        """
        if not isinstance(text, str):
            raise ValueError(f"Job description input must be a string, got {type(text).__name__}.")

        stripped = text.strip()
        if not stripped:
            raise ValueError("Job description text cannot be empty or whitespace-only.")

        return stripped

    def canonicalize_skill(self, raw_skill: str) -> Optional[str]:
        """
        Map a raw skill mention to its canonical taxonomy name.
        Returns None if the skill is unrecognized.
        """
        cleaned = raw_skill.strip().lower()
        if not cleaned:
            return None
        return SYNONYM_LOOKUP.get(cleaned)

    def process(
        self,
        raw_text: str,
        job_title_override: Optional[str] = None,
        company_override: Optional[str] = None,
    ) -> JobDescriptionProfile:
        """
        Process raw job description text into a validated JobDescriptionProfile.

        Args:
            raw_text: Raw verbatim text of the job description.
            job_title_override: Optional user-supplied job title.
            company_override: Optional user-supplied company name.

        Returns:
            JobDescriptionProfile: Fully validated and counted profile.
        """
        validated_text = self.validate_input(raw_text)
        warnings: list[str] = []

        # Enforce maximum length limit with truncation warning
        if len(validated_text) > MAX_JD_CHAR_LIMIT:
            validated_text = validated_text[:MAX_JD_CHAR_LIMIT]
            warnings.append(
                f"Job description exceeds maximum allowable length of {MAX_JD_CHAR_LIMIT:,} characters; "
                f"truncated to {MAX_JD_CHAR_LIMIT:,} characters."
            )

        # Warn on extremely short job descriptions
        word_count = len(validated_text.split())
        if word_count < 20:
            warnings.append(
                f"Job description is extremely short ({word_count} words). Criteria and requirements may be incomplete."
            )

        # Step 1: Semantic or Heuristic Extraction
        raw_extraction: Optional[_RawJDExtraction] = None
        if self.enable_llm_extraction and self.structured_engine is not None:
            try:
                raw_extraction = self._extract_with_llm(validated_text)
            except Exception as exc:
                sanitized_msg = redact_secrets(str(exc))
                warnings.append(
                    f"LLM semantic extraction failed ({type(exc).__name__}: {sanitized_msg}). "
                    f"Deterministic heuristic extraction applied."
                )

        if raw_extraction is None:
            raw_extraction = self._extract_heuristically(validated_text)

        # Step 2: Resolve Metadata Overrides
        effective_title = (
            job_title_override.strip() if job_title_override and job_title_override.strip()
            else (raw_extraction.job_title.strip() if raw_extraction.job_title else None)
        )
        effective_company = (
            company_override.strip() if company_override and company_override.strip()
            else (raw_extraction.company.strip() if raw_extraction.company else None)
        )

        # Step 3: Deterministic Normalization and Deduplication
        normalized_reqs: list[JobRequirement] = []
        seen_skill_ids: set[str] = set()

        # Track categorized skill lists
        canonical_required_skills: list[str] = []
        canonical_preferred_skills: list[str] = []

        # 3A. Process Required Skills (Precedence: Required takes priority over preferred)
        for raw_s in raw_extraction.required_skills:
            canon = self.canonicalize_skill(raw_s) or raw_s.strip()
            if not canon:
                continue
            skill_id = f"req-skill-{self._slugify(canon)}"
            if skill_id not in seen_skill_ids:
                seen_skill_ids.add(skill_id)
                canonical_required_skills.append(canon)
                normalized_reqs.append(
                    JobRequirement(
                        requirement_id=skill_id,
                        requirement_text=raw_s.strip(),
                        canonical_name=canon,
                        category=RequirementCategory.TECHNICAL_SKILL,
                        priority=RequirementPriority.REQUIRED,
                        evidence_expectation="Demonstrated proficiency in resume skills or experience",
                        source_section="Required Skills",
                    )
                )

        # 3B. Process Preferred Skills
        for raw_s in raw_extraction.preferred_skills:
            canon = self.canonicalize_skill(raw_s) or raw_s.strip()
            if not canon:
                continue
            skill_id = f"req-skill-{self._slugify(canon)}"
            if skill_id not in seen_skill_ids:
                seen_skill_ids.add(skill_id)
                canonical_preferred_skills.append(canon)
                normalized_reqs.append(
                    JobRequirement(
                        requirement_id=skill_id,
                        requirement_text=raw_s.strip(),
                        canonical_name=canon,
                        category=RequirementCategory.TECHNICAL_SKILL,
                        priority=RequirementPriority.PREFERRED,
                        evidence_expectation="Demonstrated exposure in resume skills, experience, or projects",
                        source_section="Preferred Skills",
                    )
                )

        # 3C. Process Experience Tenure Requirement
        exp_req_text = raw_extraction.experience_requirement
        if exp_req_text and exp_req_text.strip():
            normalized_reqs.append(
                JobRequirement(
                    requirement_id="req-exp-tenure",
                    requirement_text=exp_req_text.strip(),
                    canonical_name="Minimum Professional Experience",
                    category=RequirementCategory.EXPERIENCE_TENURE,
                    priority=RequirementPriority.REQUIRED,
                    evidence_expectation=exp_req_text.strip(),
                    source_section="Experience Requirements",
                )
            )

        # 3D. Process Specific Target Role Title Requirement
        if effective_title:
            normalized_reqs.append(
                JobRequirement(
                    requirement_id="req-role-title",
                    requirement_text=effective_title,
                    canonical_name=effective_title,
                    category=RequirementCategory.ROLE_RELEVANCE,
                    priority=RequirementPriority.REQUIRED,
                    evidence_expectation=f"Relevant background in {effective_title}",
                    source_section="Job Title",
                )
            )

        # 3E. Process Education Requirement
        edu_req_text = raw_extraction.education_requirement
        if edu_req_text and edu_req_text.strip():
            normalized_reqs.append(
                JobRequirement(
                    requirement_id="req-edu-level",
                    requirement_text=edu_req_text.strip(),
                    canonical_name="Academic Education Requirement",
                    category=RequirementCategory.EDUCATION,
                    priority=RequirementPriority.REQUIRED,
                    evidence_expectation=edu_req_text.strip(),
                    source_section="Education Requirements",
                )
            )

        # 3F. Process Certifications
        deduped_certs: list[str] = []
        seen_certs: set[str] = set()
        for c in raw_extraction.certifications:
            clean_c = c.strip()
            if clean_c and clean_c.lower() not in seen_certs:
                seen_certs.add(clean_c.lower())
                deduped_certs.append(clean_c)
                normalized_reqs.append(
                    JobRequirement(
                        requirement_id=f"req-cert-{self._slugify(clean_c)}",
                        requirement_text=clean_c,
                        canonical_name=clean_c,
                        category=RequirementCategory.CERTIFICATION,
                        priority=RequirementPriority.REQUIRED,
                        evidence_expectation=clean_c,
                        source_section="Certifications",
                    )
                )

        # 3G. Process Project Requirements
        deduped_proj_reqs: list[str] = []
        seen_projs: set[str] = set()
        for p in raw_extraction.project_requirements:
            clean_p = p.strip()
            if clean_p and clean_p.lower() not in seen_projs:
                seen_projs.add(clean_p.lower())
                deduped_proj_reqs.append(clean_p)
                normalized_reqs.append(
                    JobRequirement(
                        requirement_id=f"req-proj-{self._slugify(clean_p)}",
                        requirement_text=clean_p,
                        canonical_name=clean_p,
                        category=RequirementCategory.PROJECT,
                        priority=RequirementPriority.PREFERRED,
                        evidence_expectation=clean_p,
                        source_section="Project Requirements",
                    )
                )

        # 3H. Deduplicate Responsibilities
        deduped_responsibilities: list[str] = []
        seen_resps: set[str] = set()
        for r in raw_extraction.responsibilities:
            clean_r = r.strip()
            if clean_r and clean_r.lower() not in seen_resps:
                seen_resps.add(clean_r.lower())
                deduped_responsibilities.append(clean_r)

        # Step 4: Requirement Counting Policy
        n_req_skills = len(canonical_required_skills)
        n_pref_skills = len(canonical_preferred_skills)
        n_skills = n_req_skills + n_pref_skills

        n_exp_tenure = 1 if exp_req_text and exp_req_text.strip() else 0
        n_role_title = 1 if effective_title else 0
        n_edu = 1 if edu_req_text and edu_req_text.strip() else 0
        n_certs = len(deduped_certs)
        n_proj_req = 1 if deduped_proj_reqs else 0

        n_total_reqs = n_skills + n_exp_tenure + n_role_title + n_edu + n_certs + n_proj_req

        requirement_counts = {
            "required_technical_skills": n_req_skills,
            "preferred_technical_skills": n_pref_skills,
            "technical_skills": n_skills,
            "experience_tenure": n_exp_tenure,
            "role_title": n_role_title,
            "education": n_edu,
            "certifications": n_certs,
            "project_requirements": n_proj_req,
            "total_requirements": n_total_reqs,
        }

        # Step 5: Provisional Eligibility & Insufficient Evidence Check
        if n_total_reqs == 0:
            eligibility = EvaluationStatus.PROVISIONAL_EVALUATION
            warnings.append(
                "No assessable technical requirements or qualifications could be extracted from the job description. "
                "Insufficient evidence for job matching."
            )
        elif n_skills < MIN_TECHNICAL_SKILLS_THRESHOLD or n_total_reqs < MIN_TOTAL_REQUIREMENTS_THRESHOLD:
            eligibility = EvaluationStatus.PROVISIONAL_EVALUATION
            warnings.append(
                f"Sparse job description: contains only {n_skills} technical skills and "
                f"{n_total_reqs} total assessable requirements (below threshold of "
                f"{MIN_TECHNICAL_SKILLS_THRESHOLD} skills and {MIN_TOTAL_REQUIREMENTS_THRESHOLD} total requirements). "
                f"Match evaluation will be PROVISIONAL."
            )
        else:
            eligibility = EvaluationStatus.FULL_EVALUATION

        # Step 6: Assemble Immutable Profile
        return JobDescriptionProfile(
            job_title=effective_title,
            company=effective_company,
            original_description=validated_text,
            normalized_requirements=normalized_reqs,
            experience_requirement=exp_req_text,
            education_requirement=edu_req_text,
            responsibilities=deduped_responsibilities,
            required_skills=canonical_required_skills,
            preferred_skills=canonical_preferred_skills,
            certifications=deduped_certs,
            project_requirements=deduped_proj_reqs,
            requirement_counts=requirement_counts,
            evaluation_eligibility=eligibility,
            processing_warnings=warnings,
        )

    def _extract_with_llm(self, text: str) -> _RawJDExtraction:
        """
        Extract structured JD requirements via StructuredOutputEngine.
        Enforces schema validation and bounded retry guarantees.
        """
        assert self.structured_engine is not None, "structured_engine cannot be None"

        system_instruction = (
            "You are an expert technical recruiting parser. Your task is to extract qualification "
            "requirements, responsibilities, and metadata from the job description text.\n"
            "Strict Instructions:\n"
            "1. Extract ONLY information explicitly stated or directly quoted in the text.\n"
            "2. Do NOT invent missing company names, titles, or technical skills.\n"
            "3. Distinguish required mandatory skills from preferred nice-to-have skills.\n"
            "4. Return clean, unembellished lists of strings."
        )

        prompt = f"Analyze the following job description and extract requirements:\n\n{text}"
        req = LLMRequest.from_prompt(
            prompt=prompt,
            system_prompt=system_instruction,
            is_sensitive=False,  # JDs are public job postings
            allow_fallback=True,
        )

        response = self.structured_engine.generate(req, schema_class=_RawJDExtraction, max_attempts=2)
        return response.data

    def _extract_heuristically(self, text: str) -> _RawJDExtraction:
        """
        Deterministic, offline extraction using regex, section segmentation, and taxonomy scanning.
        Guarantees 100% test reproducibility with zero network requirements.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        job_title: Optional[str] = None
        company: Optional[str] = None
        experience_requirement: Optional[str] = None
        education_requirement: Optional[str] = None

        required_skills: list[str] = []
        preferred_skills: list[str] = []
        responsibilities: list[str] = []
        certifications: list[str] = []
        project_requirements: list[str] = []

        # 1. Job Title & Company Detection
        for line in lines[:8]:
            title_match = re.search(r'^(?:Job\s+Title|Title|Position|Role):\s*(.+)$', line, re.IGNORECASE)
            if title_match and not job_title:
                job_title = title_match.group(1).strip()

            company_match = re.search(r'^(?:Company|Organization|Employer):\s*(.+)$', line, re.IGNORECASE)
            if company_match and not company:
                company = company_match.group(1).strip()

        # If title not found via prefix, check common title phrases in first line
        if not job_title and lines:
            first_line = lines[0]
            colon_match = re.match(
                r'^([^:]*(?:Engineer|Developer|Architect|Scientist|Manager|Analyst|Consultant)[^:]*):',
                first_line,
                re.IGNORECASE,
            )
            if colon_match:
                job_title = colon_match.group(1).strip()
            elif len(first_line.split()) <= 6 and re.search(
                r'(?:Engineer|Developer|Architect|Scientist|Manager|Analyst|Consultant)',
                first_line,
                re.IGNORECASE,
            ):
                job_title = first_line.strip()

        # 2. Section Segmentation
        sections: dict[str, list[str]] = {
            "required": [],
            "preferred": [],
            "responsibilities": [],
            "education": [],
            "other": [],
        }
        current_section = "other"

        for line in lines:
            clean_header = re.sub(r'^[#*_\-\s\d\.\)]+', '', line).strip().lower()
            clean_header = re.sub(r'[:*_\s]+$', '', clean_header)

            req_match = re.match(
                r'^(?:requirements?|qualifications?|must\s+have|what\s+you(?:\'ll)?\s+need|minimum\s+qualifications?|basic\s+qualifications?|who\s+you\s+are|what\s+(?:we\s+are|we\'re|we)\s+looking\s+for|skills?\s*(?:&|and)\s*experience|core\s+competencies|tech\s+stack|technical\s+requirements?|key\s+requirements?|what\s+you\s+bring|candidate\s+profile|ideal\s+candidate|about\s+you|what\s+you\s+need\s+to\s+bring|builder\s+background|builder\s+profile|requirements?\s*(?:&|and)\s*candidate\s+profile|what\s+we\s+expect|profile\s*(?:&|and)\s*expectations?)\b',
                clean_header,
            )
            pref_match = re.match(
                r'^(?:preferred|nice\s+to\s+have|bonus|desired|preferred\s+qualifications?|preferred\s+skills?|good\s+to\s+have|additional\s+skills?|plus|bonus\s+points?|what\s+gives\s+you\s+an\s+edge)\b',
                clean_header,
            )
            resp_match = re.match(
                r'^(?:responsibilities|duties|what\s+you(?:\'ll)?\s+do|what\s+you\s+will\s+do|key\s+responsibilities|role\s+overview|about\s+the\s+role|day\s+to\s+day|what\s+you(?:\'ll)?\s+be\s+doing|the\s+role|what\s+you(?:\'ll)?\s+be\s+building|what\s+you\s+will\s+build|what\s+we\s+are\s+building|the\s+opportunity|scope\s+of\s+work|work\s+ai-first)\b',
                clean_header,
            )
            edu_match_sec = re.match(
                r'^(?:education|academic\s+background|educational\s+qualifications?|degree\s+requirements?)\b',
                clean_header,
            )

            if req_match:
                current_section = "required"
                rem = clean_header[req_match.end():].strip()
                if rem:
                    sections[current_section].append(rem)
                continue
            elif pref_match:
                current_section = "preferred"
                rem = clean_header[pref_match.end():].strip()
                if rem:
                    sections[current_section].append(rem)
                continue
            elif resp_match:
                current_section = "responsibilities"
                rem = clean_header[resp_match.end():].strip()
                if rem:
                    sections[current_section].append(rem)
                continue
            elif edu_match_sec:
                current_section = "education"
                rem = clean_header[edu_match_sec.end():].strip()
                if rem:
                    sections[current_section].append(rem)
                continue

            sections[current_section].append(line)

        # 3. Experience Detection
        # Captures single years, ranges (e.g. "1–2", "1-2", "3 to 5"), plus-suffixes ("3+"),
        # and domain modifiers up to 60 characters (e.g. "professional backend engineering experience")
        # while preventing false positives against dates, percentages, or transaction counts.
        exp_match = re.search(
            r'(\b\d+\+?(?:\s*(?:[\-\u2013\u2014]|to)\s*\d+\+?)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[a-z\s,\/\-]{0,60})?\s*experience\b)',
            text,
            re.IGNORECASE,
        )
        if exp_match:
            experience_requirement = exp_match.group(1).strip()

        # 4. Education Detection
        edu_match = re.search(
            r'\b((?:Bachelor[\'’]?s?|Master[\'’]?s?|Ph\.?D\.?|B\.?S\.?|M\.?S\.?|Associate[\'’]?s?)(?:\s+degree)?(?:\s+in\s+[\w\s,\/\-]+)?)\b',
            text,
            re.IGNORECASE,
        )
        if edu_match:
            education_requirement = edu_match.group(1).strip()

        # 5. Project & Builder Requirement Detection
        proj_match = re.search(
            r'((?:demonstrated|proven|portfolio|github|practical)\s+(?:of\s+)?(?:projects?|code\s+samples?|repositories|portfolio)(?:[\w\s]{0,40})?)',
            text,
            re.IGNORECASE,
        )
        if proj_match:
            project_requirements.append(proj_match.group(1).strip())

        builder_match = re.search(
            r'((?:0-to-1|zero-to-one|startup|side\s+projects?|builder\s+background|founding\s+team|shipped\s+(?:products?|apps?|solutions?))[\w\s,\(\)\-]{0,60})',
            text,
            re.IGNORECASE,
        )
        if builder_match and builder_match.group(1).strip() not in project_requirements:
            project_requirements.append(builder_match.group(1).strip())

        # 6. Certification Detection
        cert_matches = re.findall(
            r'\b((?:AWS\s+Certified|CKA|CKAD|PMP|CISSP|GCP\s+Professional|Azure\s+Solutions\s+Architect)[\w\s\-]*)\b',
            text,
            re.IGNORECASE,
        )
        for cm in cert_matches:
            certifications.append(cm.strip())

        # 7. Taxonomy-Driven Skill Extraction
        def _scan_skills_in_lines(target_lines: list[str]) -> list[str]:
            found_skills: list[str] = []
            block_text = " ".join(target_lines)
            for canon_name, (_, aliases) in SKILL_TAXONOMY.items():
                all_tokens = {canon_name} | aliases
                for token in all_tokens:
                    tok_lower = token.lower()
                    if tok_lower in ("c", "r"):
                        # Strictly case-sensitive for single-letter programming languages C and R
                        pattern = rf'(?<!\w){re.escape(token)}(?!\w)'
                        if re.search(pattern, block_text):
                            found_skills.append(canon_name)
                            break
                    elif tok_lower in ("go", "js", "ts", "py"):
                        # Case-insensitive word boundary for 2-letter tech tokens (reject hyphenated words like go-to)
                        pattern = rf'\b{re.escape(token)}\b(?![\w\-])'
                        if re.search(pattern, block_text, re.IGNORECASE):
                            found_skills.append(canon_name)
                            break
                    else:
                        pattern = rf'(?<!\w){re.escape(token)}(?![\w\-])'
                        if re.search(pattern, block_text, re.IGNORECASE):
                            found_skills.append(canon_name)
                            break
            return found_skills

        # Required skills from "required" section
        req_scanned = _scan_skills_in_lines(sections["required"])
        # Preferred skills from "preferred" section
        pref_scanned = _scan_skills_in_lines(sections["preferred"])
        # Skills from responsibilities
        resp_scanned = _scan_skills_in_lines(sections["responsibilities"])
        # Skills from other / tech stack sections
        other_scanned = _scan_skills_in_lines(sections["other"])

        # Prioritization & Deduplication:
        for s in req_scanned:
            if s not in required_skills:
                required_skills.append(s)

        for s in pref_scanned:
            if s not in required_skills and s not in preferred_skills:
                preferred_skills.append(s)

        # Technical skills found in responsibilities or tech stack
        for s in resp_scanned + other_scanned:
            if s not in required_skills and s not in preferred_skills:
                required_skills.append(s)

        # If still sparse (< MIN_TECHNICAL_SKILLS_THRESHOLD), scan entire text
        if len(required_skills) + len(preferred_skills) < MIN_TECHNICAL_SKILLS_THRESHOLD:
            all_scanned = _scan_skills_in_lines(lines)
            for s in all_scanned:
                if s not in required_skills and s not in preferred_skills:
                    required_skills.append(s)

        # 8. Responsibilities Extraction
        resp_lines = sections["responsibilities"]
        for r_line in resp_lines:
            # Clean leading bullet markers
            cleaned_bullet = re.sub(r'^[•\-\*\d\.\)\s]+', '', r_line).strip()
            if len(cleaned_bullet) >= 15:
                responsibilities.append(cleaned_bullet)

        # Fallback bullet extraction if responsibilities section was empty
        if not responsibilities:
            for line in lines:
                if re.match(r'^[•\-\*]\s+', line):
                    cleaned_b = re.sub(r'^[•\-\*\d\.\)\s]+', '', line).strip()
                    if len(cleaned_b) >= 20 and not any(k in cleaned_b.lower() for k in ["degree in", "years of experience", "years of professional", "bachelor", "master", "years of"]):
                        responsibilities.append(cleaned_b)

        return _RawJDExtraction(
            job_title=job_title,
            company=company,
            experience_requirement=experience_requirement,
            education_requirement=education_requirement,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            responsibilities=responsibilities,
            certifications=certifications,
            project_requirements=project_requirements,
        )

    @staticmethod
    def _slugify(text: str) -> str:
        """Create a clean, deterministic slug for requirement IDs."""
        s = re.sub(r'[^\w\s\-]', '', text).strip().lower()
        return re.sub(r'[\s\-]+', '-', s)


# Convenience Functional Wrapper
def process_job_description(
    raw_text: str,
    job_title: Optional[str] = None,
    company: Optional[str] = None,
    structured_engine: Optional[StructuredOutputEngine] = None,
) -> JobDescriptionProfile:
    """Convenience helper function to parse raw JD text."""
    processor = JobDescriptionProcessor(structured_engine=structured_engine)
    return processor.process(
        raw_text=raw_text,
        job_title_override=job_title,
        company_override=company,
    )
