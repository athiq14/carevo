"""
Deterministic education evaluator for candidate-job matching.
Evaluates academic degrees, degree levels, and discipline relevance.
Prevents false major requirements when employers specify generic degree levels only.
"""
import re
from typing import Optional

from models.matching import (
    EvidenceStatus,
    JobDescriptionProfile,
)
from models.resume_entities import EducationRecord
from models.resume_profile import ResumeProfile


# Core and Adjacent STEM Discipline Categories
CORE_STEM_FIELDS = {
    "computer science",
    "software engineering",
    "computer engineering",
    "data science",
    "artificial intelligence",
    "machine learning",
    "information technology",
    "computational science",
}

ADJACENT_STEM_FIELDS = {
    "information systems",
    "electrical engineering",
    "electronics engineering",
    "mathematics",
    "applied mathematics",
    "statistics",
    "physics",
    "cybersecurity",
    "telecommunications",
}

DEGREE_LEVEL_RANKS = {
    "phd": 4,
    "ph.d": 4,
    "doctorate": 4,
    "master": 3,
    "m.s.": 3,
    "ms": 3,
    "msc": 3,
    "bachelor": 2,
    "b.s.": 2,
    "bs": 2,
    "bsc": 2,
    "b.e.": 2,
    "b.tech": 2,
    "associate": 1,
    "diploma": 1,
    "certificate": 1,
}


class EducationEvaluator:
    """
    Deterministic evaluator for candidate academic qualifications.
    Ensures discipline relevance is evaluated only when explicitly requested by the employer.
    """

    def evaluate(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
    ) -> dict[str, object]:
        """
        Evaluate candidate education against job description requirements.

        Returns:
            dict containing:
                - dimension_score: float bounded [0.0, 100.0]
                - level_score: Optional[float]
                - major_score: Optional[float]
                - is_active: bool
                - neutralization_explanation: Optional[str]
                - evidence_status: EvidenceStatus
                - diagnostics: list[str]
        """
        diagnostics: list[str] = []
        edu_req = jd_profile.education_requirement

        # Case 0: No education requirement specified in JD -> Neutralize dimension
        if not edu_req or not edu_req.strip():
            return {
                "dimension_score": 0.0,
                "level_score": None,
                "major_score": None,
                "is_active": False,
                "neutralization_explanation": "Job description specifies no academic education criteria.",
                "evidence_status": EvidenceStatus.UNKNOWN_EVIDENCE,
                "diagnostics": ["No academic qualification requirement stated in job description."],
            }

        # Step 1: Analyze candidate's highest academic qualification
        highest_record, cand_rank = self._find_highest_qualification(resume_profile.education)

        if not highest_record or cand_rank == 0:
            # Candidate has no educational records in resume -> MISSING_EVIDENCE
            return {
                "dimension_score": 0.0,
                "level_score": 0.0,
                "major_score": 0.0,
                "is_active": True,
                "neutralization_explanation": None,
                "evidence_status": EvidenceStatus.MISSING_EVIDENCE,
                "diagnostics": ["No academic qualifications found in candidate resume."],
            }

        # Step 2: Determine JD Degree Level Expectation
        jd_req_lower = edu_req.lower()
        req_rank = self._parse_required_rank(jd_req_lower)

        # Calculate Level Sub-score
        level_score: float
        if cand_rank >= req_rank:
            level_score = 100.0
            diagnostics.append(
                f"Candidate degree ({highest_record.degree or 'Degree'}) satisfies requested degree level (rank {cand_rank} >= {req_rank})."
            )
        elif cand_rank == req_rank - 1:
            level_score = 50.0
            diagnostics.append(
                f"Candidate degree rank ({cand_rank}) is one tier below requested level ({req_rank}): awarded 50.0% level credit."
            )
        else:
            level_score = 25.0
            diagnostics.append(
                f"Candidate degree rank ({cand_rank}) is below requested level ({req_rank}): awarded 25.0% level credit."
            )

        # Step 3: Check whether JD specifies a discipline or major requirement
        specifies_major, requested_major = self._check_discipline_requirement(jd_req_lower)

        if not specifies_major:
            # Generic degree level requirement only (e.g. "Bachelor's degree required")
            # Major is NOT evaluated; any recognized degree at requested level earns 100%
            diagnostics.append(
                "Job description specifies a generic degree level without discipline restriction; major relevance not evaluated."
            )
            dimension_score = level_score
            ev_status = EvidenceStatus.VERIFIED_EVIDENCE if dimension_score >= 100.0 else EvidenceStatus.PARTIAL_EVIDENCE

            return {
                "dimension_score": dimension_score,
                "level_score": level_score,
                "major_score": None,
                "is_active": True,
                "neutralization_explanation": None,
                "evidence_status": ev_status,
                "diagnostics": diagnostics,
            }

        # Step 4: Discipline-Specific Evaluation (60% Level + 40% Major)
        field_of_study = (highest_record.field_of_study or "").lower().strip()
        if not field_of_study:
            # Try to extract major from degree text (e.g., "Bachelor of Science in Computer Science")
            deg_text = (highest_record.degree or "").lower()
            m = re.search(r'\bin\s+([\w\s]+)', deg_text)
            if m:
                field_of_study = m.group(1).strip()

        major_score: float
        if not field_of_study:
            # Field of study unspecified in resume -> UNKNOWN_EVIDENCE (neutralize major sub-component)
            diagnostics.append("Candidate academic major unspecified in resume; major sub-component neutralized.")
            dimension_score = level_score
            ev_status = EvidenceStatus.VERIFIED_EVIDENCE if dimension_score >= 100.0 else EvidenceStatus.PARTIAL_EVIDENCE
            return {
                "dimension_score": dimension_score,
                "level_score": level_score,
                "major_score": None,
                "is_active": True,
                "neutralization_explanation": None,
                "evidence_status": ev_status,
                "diagnostics": diagnostics,
            }

        # Evaluate candidate's major alignment against requested discipline
        if any(c in field_of_study for c in CORE_STEM_FIELDS):
            major_score = 100.0
            diagnostics.append(f"Candidate major '{highest_record.field_of_study}' matches Core STEM discipline: 100.0%.")
        elif any(a in field_of_study for a in ADJACENT_STEM_FIELDS):
            major_score = 75.0
            diagnostics.append(f"Candidate major '{highest_record.field_of_study}' matches Adjacent STEM discipline: 75.0%.")
        else:
            major_score = 25.0
            diagnostics.append(f"Candidate major '{highest_record.field_of_study}' is outside requested STEM disciplines: 25.0%.")

        dimension_score = 0.60 * level_score + 0.40 * major_score
        dimension_score = max(0.0, min(100.0, dimension_score))
        ev_status = EvidenceStatus.VERIFIED_EVIDENCE if dimension_score >= 80.0 else EvidenceStatus.PARTIAL_EVIDENCE

        return {
            "dimension_score": dimension_score,
            "level_score": level_score,
            "major_score": major_score,
            "is_active": True,
            "neutralization_explanation": None,
            "evidence_status": ev_status,
            "diagnostics": diagnostics,
        }

    def _find_highest_qualification(self, education_records: list[EducationRecord]) -> tuple[Optional[EducationRecord], int]:
        """Find the candidate's highest educational degree and its rank."""
        best_record: Optional[EducationRecord] = None
        best_rank = 0

        for rec in education_records:
            full_text = f"{rec.degree or ''} {rec.original_text}".lower()
            rank = 0
            for term, r in DEGREE_LEVEL_RANKS.items():
                if term in full_text:
                    if r > rank:
                        rank = r
            if rank > best_rank:
                best_rank = rank
                best_record = rec
            elif best_record is None:
                best_record = rec

        return best_record, best_rank

    def _parse_required_rank(self, req_text: str) -> int:
        """Parse required degree rank from JD text. Defaults to Bachelor's (rank 2)."""
        if any(t in req_text for t in ("phd", "ph.d", "doctorate", "doctoral")):
            return 4
        if any(t in req_text for t in ("master", "m.s.", "ms", "msc")):
            return 3
        if any(t in req_text for t in ("bachelor", "b.s.", "bs", "bsc", "degree", "undergraduate")):
            return 2
        if any(t in req_text for t in ("associate", "diploma")):
            return 1
        return 2  # Standard professional baseline

    def _check_discipline_requirement(self, req_text: str) -> tuple[bool, Optional[str]]:
        """Check whether JD text explicitly requests a specific discipline (e.g. CS, STEM)."""
        stem_indicators = (
            "computer science",
            "stem",
            "software engineering",
            "electrical engineering",
            "information technology",
            "mathematics",
            "related field",
            "related technical discipline",
            "related discipline",
        )
        for ind in stem_indicators:
            if ind in req_text:
                return True, ind
        return False, None

