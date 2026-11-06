"""
Deterministic project evaluator for candidate-job matching.
Evaluates practical candidate exposure across three objective pillars:
Pillar A: Technology Coverage & Application (max 40 pts)
Pillar B: Contextual Depth & Verification (max 35 pts)
Pillar C: Responsibility Alignment (max 25 pts)
Strictly enforces artifact verification, quantified metric standards, and toolchain workflow criteria.
"""
import re
from typing import Optional
from urllib.parse import urlsplit

from models.matching import (
    EvidenceStatus,
    EvidenceVerificationLevel,
    JobDescriptionProfile,
    ProjectEvidence,
)
from models.resume_entities import ProjectRecord
from models.resume_profile import ResumeProfile
from processors.entity_extractor import URL_EXCLUSION_PATTERN
from processors.jd_processor import SYNONYM_LOOKUP
from core.matching.skill_matcher import RELATED_TECH_MAP


# Recognized code repository and artifact hosting domains
ACCEPTED_ARTIFACT_HOSTS = {
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "huggingface.co",
    "kaggle.com",
    "codeberg.org",
    "sourceforge.net",
}


def _is_accepted_artifact_host(hostname: str) -> bool:
    """Verify if hostname is an accepted code repository or artifact domain."""
    clean_host = hostname.lower().strip()
    if clean_host in ACCEPTED_ARTIFACT_HOSTS:
        return True
    if clean_host.endswith(".github.io") or clean_host.endswith(".gitlab.io"):
        parts = clean_host.split(".")
        if len(parts) >= 3 and parts[0]:
            return True
    return False


def validate_artifact_url(url: Optional[str]) -> tuple[bool, EvidenceVerificationLevel, str]:
    """
    Validate a candidate-provided project artifact URL offline without network requests.
    Returns (is_valid_for_points, verification_level, explanation).

    Offline matching establishes:
    - URL syntax is valid.
    - Scheme is secure HTTPS.
    - Hostname belongs to an accepted artifact or portfolio domain.
    - URL path specifies a repository/artifact.

    Offline matching does not check remote accessibility or repository contents.
    """
    if not url or not url.strip():
        return False, EvidenceVerificationLevel.SELF_REPORTED, "No artifact URL provided."

    clean_url = url.strip()
    try:
        parsed = urlsplit(clean_url)
    except Exception as exc:
        return False, EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL, f"Malformed URL: {exc}"

    # Scheme check: must be https
    scheme = parsed.scheme.lower() if parsed.scheme else ""
    if scheme != "https":
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"Unsupported scheme '{scheme or 'none'}'; secure HTTPS is required for artifact verification.",
        )

    # Credentials check: username and password must both be None
    try:
        if parsed.username is not None or parsed.password is not None:
            return (
                False,
                EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
                "Embedded username or password credentials in artifact URL are forbidden.",
            )
    except Exception as exc:
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"Malformed credentials in URL: {exc}",
        )

    # Port check: only 443 or default (None) is acceptable
    netloc_host = parsed.netloc.split("@")[-1]
    if ":" in netloc_host:
        port_part = netloc_host.rsplit(":", 1)[1]
        if not port_part:
            return (
                False,
                EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
                "Malformed port in URL: empty port following colon.",
            )

    try:
        port = parsed.port
    except ValueError as exc:
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"Malformed port in URL: {exc}",
        )
    except Exception as exc:
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"Invalid port specification in URL: {exc}",
        )

    if port is not None and port != 443:
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"Unexpected port '{port}'; standard HTTPS port 443 (or default) is required.",
        )

    # Hostname check: must be present and accepted
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL, "URL is missing a valid hostname."

    if not _is_accepted_artifact_host(hostname):
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"Hostname '{hostname}' is not in the recognized artifact or portfolio domain list.",
        )

    # Path check: must have non-empty path beyond '/'
    clean_path = parsed.path.strip("/")
    if not clean_path:
        return (
            False,
            EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL,
            f"URL path is empty for host '{hostname}'; repository or artifact path is required.",
        )

    return (
        True,
        EvidenceVerificationLevel.STRUCTURALLY_VALID_URL,
        (
            f"Structurally valid HTTPS artifact URL on recognized host '{hostname}'. "
            "Offline matching verifies syntax and host domain only; remote repository accessibility, "
            "contents, and authorship are not inspected."
        ),
    )


# Regex for quantified performance or throughput metric clauses
QUANTIFIED_METRIC_PATTERN = re.compile(
    r'(?:\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|s|tps|qps|rps|reqs?\/sec|users?|gb|tb|mb|concurrent)\b|\breduced\s+[\w\s]{0,20}\s+by\s+\d+|\bfrom\s+\d+(?:\.\d+)?\w*\s+to\s+\d+(?:\.\d+)?\w*)',
    re.IGNORECASE,
)

# Regex for operational interactive workflow between tools
TOOLCHAIN_WORKFLOW_PATTERNS = [
    re.compile(r'querying\s+[\w\s\-]+(?:\s+through|\s+using|\s+via)', re.IGNORECASE),
    re.compile(r'containerized\s+with\s+[\w\s\-]+', re.IGNORECASE),
    re.compile(r'deployed\s+(?:on|to|via)\s+[\w\s\-]+', re.IGNORECASE),
    re.compile(r'pipeline\s+[\w\s\-]+(?:streaming|ingesting|processing)\s+(?:from|to|via)', re.IGNORECASE),
    re.compile(r'integrated\s+with\s+[\w\s\-]+(?:and|using)', re.IGNORECASE),
    re.compile(r'asynchronous\s+connection\s+pool', re.IGNORECASE),
]


# Domain concept clusters for duty and responsibility alignment
DUTY_DOMAIN_MAP = {
    'database': {'database', 'relational', 'schema', 'schemas', 'query', 'queries', 'querying', 'sql', 'tables', 'table', 'postgresql', 'mysql', 'hibernate'},
    'api': {'api', 'apis', 'rest', 'restful', 'endpoints', 'service', 'services', 'microservice', 'microservices', 'fastapi', 'flask', 'django', 'spring boot'},
    'cloud': {'cloud', 'infrastructure', 'terraform', 'kubernetes', 'k8s', 'aws', 'gcp', 'azure', 'devops', 'ci/cd', 'deployment'},
    'software': {'software', 'web', 'application', 'applications', 'code', 'develop', 'modular', 'auth', 'optimization', 'querying', 'pipeline'},
}


class ProjectEvaluator:
    """
    Deterministic evaluator for engineering and technical projects.
    Awards compatibility credit only for verified relevant engineering.
    """

    def __init__(
        self,
        synonym_lookup: dict[str, str] = SYNONYM_LOOKUP,
        related_tech_map: dict[str, dict[str, float]] = RELATED_TECH_MAP,
    ) -> None:
        self.synonym_lookup = synonym_lookup
        self.related_tech_map = related_tech_map

    def evaluate(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
        candidate_ft_years: float = 0.0,
    ) -> dict[str, object]:
        """
        Evaluate candidate project portfolio against job description criteria.

        Returns:
            dict containing:
                - dimension_score: float bounded [0.0, 100.0]
                - c_tech: float (Pillar A, max 40.0)
                - d_context: float (Pillar B, max 35.0)
                - v_align: float (Pillar C, max 25.0)
                - project_evidence: list[ProjectEvidence]
                - relevant_projects_count: int
                - is_active: bool
                - neutralization_explanation: Optional[str]
                - evidence_status: EvidenceStatus
                - diagnostics: list[str]
        """
        diagnostics: list[str] = []

        # Check if JD specifies no project requirements and candidate has no projects:
        has_jd_proj_req = bool(jd_profile.project_requirements) or bool(
            re.search(r'\b(?:project|portfolio|github|repository|code sample)\b', jd_profile.original_description, re.IGNORECASE)
        )
        has_candidate_projects = len(resume_profile.projects) > 0

        if not has_jd_proj_req and not has_candidate_projects:
            diagnostics.append("Job description specifies no project requirements and candidate provided no projects; Project dimension neutralized.")
            return {
                "dimension_score": 0.0,
                "c_tech": 0.0,
                "d_context": 0.0,
                "v_align": 0.0,
                "project_evidence": [],
                "relevant_projects_count": 0,
                "is_active": False,
                "neutralization_explanation": "Job description specifies no project or portfolio requirements.",
                "evidence_status": EvidenceStatus.UNKNOWN_EVIDENCE,
                "diagnostics": diagnostics,
            }

        # Step 1: Identify Relevant Projects (P_rel)
        rel_projects, audit_records = self._filter_and_audit_projects(
            resume_profile.projects, jd_profile
        )

        # Invariant: If candidate has no relevant projects, S_proj = 0.0
        if not rel_projects:
            diagnostics.append("Candidate projects have zero overlap with JD technologies or duties: 0.0 compatibility points.")
            return {
                "dimension_score": 0.0,
                "c_tech": 0.0,
                "d_context": 0.0,
                "v_align": 0.0,
                "project_evidence": audit_records,
                "relevant_projects_count": 0,
                "is_active": True,
                "neutralization_explanation": None,
                "evidence_status": EvidenceStatus.MISSING_EVIDENCE,
                "diagnostics": diagnostics,
            }

        # Step 2: Pillar A — Technology Coverage & Application (C_tech, max 40 points)
        c_tech = self._calculate_pillar_a(rel_projects, jd_profile, diagnostics)

        # Step 3: Pillar B — Contextual Depth (D_context, max 35 points)
        d_context = self._calculate_pillar_b(rel_projects, diagnostics)

        # Step 4: Pillar C — Duty Alignment (V_align, max 25 points)
        v_align = self._calculate_pillar_c(rel_projects, jd_profile, diagnostics)

        # Step 5: Total Project Score strictly bounded [0.0, 100.0]
        dimension_score = c_tech + d_context + v_align
        dimension_score = max(0.0, min(100.0, dimension_score))

        ev_status = EvidenceStatus.VERIFIED_EVIDENCE if dimension_score >= 70.0 else EvidenceStatus.PARTIAL_EVIDENCE

        # Attach calculated pillar scores to audit records
        for rec in audit_records:
            rec.calculated_pillar_scores["c_tech"] = c_tech
            rec.calculated_pillar_scores["d_context"] = d_context
            rec.calculated_pillar_scores["v_align"] = v_align
            rec.calculated_pillar_scores["s_proj"] = dimension_score

        return {
            "dimension_score": dimension_score,
            "c_tech": c_tech,
            "d_context": d_context,
            "v_align": v_align,
            "project_evidence": audit_records,
            "relevant_projects_count": len(rel_projects),
            "is_active": True,
            "neutralization_explanation": None,
            "evidence_status": ev_status,
            "diagnostics": diagnostics,
        }

    def _filter_and_audit_projects(
        self,
        projects: list[ProjectRecord],
        jd_profile: JobDescriptionProfile,
    ) -> tuple[list[ProjectRecord], list[ProjectEvidence]]:
        """Filter projects for JD relevance and construct formal ProjectEvidence audit records."""
        jd_skills_canonical_lower: dict[str, str] = {}
        for s in jd_profile.required_skills + jd_profile.preferred_skills:
            c = self.synonym_lookup.get(s.strip().lower(), s.strip())
            jd_skills_canonical_lower[c.lower()] = c

        # Also include related technologies
        related_lower: dict[str, str] = {}
        for canon, neighbors in self.related_tech_map.items():
            if canon.lower() in jd_skills_canonical_lower:
                for neigh in neighbors:
                    related_lower[neigh.lower()] = canon

        rel_projects: list[ProjectRecord] = []
        audit_records: list[ProjectEvidence] = []

        for idx, proj in enumerate(projects):
            matched_techs: list[str] = []
            for t in proj.technologies:
                t_clean = t.strip()
                t_canon = self.synonym_lookup.get(t_clean.lower(), t_clean)
                if t_canon.lower() in jd_skills_canonical_lower:
                    matched_techs.append(t_canon)
                elif t_canon.lower() in related_lower:
                    matched_techs.append(t_canon)

            proj_text = f"{proj.project_name} {proj.description}".lower()

            # Check for duty alignment keywords
            is_relevant = len(matched_techs) > 0 or any(
                r.lower() in proj_text for r in jd_profile.responsibilities if len(r) > 10
            )

            if is_relevant:
                rel_projects.append(proj)

            # Verification level from offline structural validation
            is_valid, verif_level, val_reason = validate_artifact_url(proj.project_url)

            # Find metrics clauses in narrative
            metric_clauses = QUANTIFIED_METRIC_PATTERN.findall(proj.description)

            audit_rec = ProjectEvidence(
                project_identifier=f"proj-{idx + 1}-{proj.project_name.lower().replace(' ', '-')[:25]}",
                project_name=proj.project_name,
                relevant_technologies=matched_techs,
                technology_implementation_evidence=[f"Implemented {', '.join(matched_techs)}"] if matched_techs else [],
                artifact_url=proj.project_url,
                evidence_verification_level=verif_level,
                self_reported_metrics=metric_clauses,
                toolchain_interaction_description=proj.description[:200] if len(proj.technologies) >= 3 else None,
                responsibility_alignment_evidence=[proj.description[:150]] if is_relevant else [],
                calculated_pillar_scores={},
            )
            audit_records.append(audit_rec)

        return rel_projects, audit_records

    def _calculate_pillar_a(
        self,
        rel_projects: list[ProjectRecord],
        jd_profile: JobDescriptionProfile,
        diagnostics: list[str],
    ) -> float:
        """
        Pillar A: Technology Coverage & Application (max 40 points).
        Evaluates implementation frequency and related technology factors.
        """
        req_skills: list[str] = []
        seen_req: set[str] = set()
        for s in jd_profile.required_skills:
            c = self.synonym_lookup.get(s.lower(), s)
            if c.lower() not in seen_req:
                seen_req.add(c.lower())
                req_skills.append(c)

        pref_skills: list[str] = []
        for s in jd_profile.preferred_skills:
            c = self.synonym_lookup.get(s.lower(), s)
            if c.lower() not in seen_req and c.lower() not in [p.lower() for p in pref_skills]:
                pref_skills.append(c)

        denominator = len(req_skills) * 1.0 + len(pref_skills) * 0.5
        if denominator <= 0.0:
            return 40.0

        numerator = 0.0

        for skill in req_skills:
            credit = self._get_skill_project_credit(skill, rel_projects)
            numerator += credit * 1.0

        for skill in pref_skills:
            credit = self._get_skill_project_credit(skill, rel_projects)
            numerator += credit * 0.5

        c_tech = 40.0 * (numerator / denominator)
        diagnostics.append(f"Pillar A (Technology Coverage): {numerator:.2f} / {denominator:.2f} -> {c_tech:.2f} points.")
        return min(40.0, max(0.0, c_tech))

    def _get_skill_project_credit(self, skill: str, rel_projects: list[ProjectRecord]) -> float:
        """
        Calculate credit for a skill across relevant projects:
        >= 2 projects: 1.0; 1 project: 0.6; related tech: factor * (1.0 or 0.6).
        """
        canonical_target = self.synonym_lookup.get(skill.strip().lower(), skill.strip())
        target_lower = canonical_target.lower()

        # 1. Direct Implementation Count
        direct_count = 0
        for p in rel_projects:
            techs_canon_lower = [
                self.synonym_lookup.get(t.strip().lower(), t.strip()).lower()
                for t in p.technologies
            ]
            desc_lower = p.description.lower()
            if target_lower in techs_canon_lower or re.search(rf'\b{re.escape(target_lower)}\b', desc_lower):
                direct_count += 1

        if direct_count >= 2:
            return 1.0
        elif direct_count == 1:
            return 0.6

        # 2. Related Technology Check
        neighbors = self.related_tech_map.get(canonical_target, {})
        for neigh, factor in neighbors.items():
            neigh_lower = neigh.lower()
            neigh_count = 0
            for p in rel_projects:
                techs_canon_lower = [
                    self.synonym_lookup.get(t.strip().lower(), t.strip()).lower()
                    for t in p.technologies
                ]
                desc_lower = p.description.lower()
                if neigh_lower in techs_canon_lower or re.search(rf'\b{re.escape(neigh_lower)}\b', desc_lower):
                    neigh_count += 1
            if neigh_count >= 2:
                return 1.0 * factor
            elif neigh_count == 1:
                return 0.6 * factor

        return 0.0

    def _calculate_pillar_b(
        self,
        rel_projects: list[ProjectRecord],
        diagnostics: list[str],
    ) -> float:
        """
        Pillar B: Contextual Depth (max 35 points).
        - B1: Artifact URL (10 points)
        - B2: Quantified Metrics (15 points if >= 2 distinct clauses, else 0)
        - B3: Toolchain Interaction Workflow (10 points if >= 3 tools interact, else 0)
        """
        # B1: Candidate-provided structurally valid artifact URL (10 points)
        valid_urls: list[str] = []
        for p in rel_projects:
            is_valid, v_level, reason = validate_artifact_url(p.project_url)
            if is_valid and v_level == EvidenceVerificationLevel.STRUCTURALLY_VALID_URL:
                valid_urls.append(p.project_url.strip())

        if valid_urls:
            b1_pts = 10.0
            diagnostics.append(
                f"Candidate provided {len(valid_urls)} structurally valid HTTPS artifact URL(s) on accepted domain(s): awarded 10.0 points for Pillar B1. "
                "Remote accessibility and repository contents were not inspected (offline evaluation)."
            )
        else:
            b1_pts = 0.0
            diagnostics.append(
                "No structurally valid HTTPS artifact URL provided on accepted domains: 0.0 points for Pillar B1."
            )

        # B2: Quantified Metrics (15 points only if >= 2 distinct clauses)
        metric_clauses: list[str] = []
        for p in rel_projects:
            matches = QUANTIFIED_METRIC_PATTERN.findall(p.description)
            for m in matches:
                clean_m = m.strip().lower()
                if clean_m not in [mc.lower() for mc in metric_clauses]:
                    metric_clauses.append(clean_m)

        b2_pts = 15.0 if len(metric_clauses) >= 2 else 0.0
        if len(metric_clauses) == 1:
            diagnostics.append("Only 1 quantified metric clause found in projects (requires >= 2): awarded 0.0 points.")

        # B3: Toolchain Interaction Workflow (10 points)
        has_toolchain = False
        for p in rel_projects:
            if len(p.technologies) >= 3:
                # Check narrative for operational interactive workflow
                desc = p.description
                if any(pat.search(desc) for pat in TOOLCHAIN_WORKFLOW_PATTERNS):
                    has_toolchain = True
                    break

        b3_pts = 10.0 if has_toolchain else 0.0
        if not has_toolchain and any(len(p.technologies) >= 3 for p in rel_projects):
            diagnostics.append(
                "Project lists >= 3 tools, but narrative omits operational interaction workflow: awarded 0.0 toolchain points."
            )

        d_context = b1_pts + b2_pts + b3_pts
        diagnostics.append(
            f"Pillar B (Contextual Depth): URL={b1_pts:.0f}, Metrics={b2_pts:.0f}, Toolchain={b3_pts:.0f} -> {d_context:.2f} points."
        )
        return min(35.0, max(0.0, d_context))

    def _calculate_pillar_c(
        self,
        rel_projects: list[ProjectRecord],
        jd_profile: JobDescriptionProfile,
        diagnostics: list[str],
    ) -> float:
        """
        Pillar C: Responsibility Alignment (max 25 points).
        - Direct alignment with >= 2 core duties: 25 points
        - Moderate alignment (1 duty match): 10 points
        - General technical execution: 5 points
        - Unrelated: 0 points
        """
        if not rel_projects:
            return 0.0

        duties = jd_profile.responsibilities
        if not duties:
            # If JD has no explicit responsibility bullets, evaluate general execution
            diagnostics.append("JD specifies no explicit responsibility bullets: awarded 5.0 general execution points.")
            return 5.0

        duty_matches = 0
        all_proj_text = " ".join(f"{p.project_name} {p.description}" for p in rel_projects).lower()
        all_proj_techs = {t.lower() for p in rel_projects for t in p.technologies}

        for d in duties:
            d_clean = d.lower()
            matched_this_duty = False

            # 1. Concept cluster check
            for domain, terms in DUTY_DOMAIN_MAP.items():
                if any(t in d_clean for t in terms):
                    if any(t in all_proj_text for t in terms) or any(t in all_proj_techs for t in terms):
                        matched_this_duty = True
                        break

            # 2. Keyword overlap fallback
            if not matched_this_duty:
                keywords = [w for w in re.findall(r'\b\w{4,}\b', d_clean) if w not in ("with", "from", "that", "this", "your")]
                if keywords and sum(1 for kw in keywords if kw in all_proj_text) >= 2:
                    matched_this_duty = True

            if matched_this_duty:
                duty_matches += 1

        if duty_matches >= 2:
            v_align = 25.0
            diagnostics.append(f"Pillar C (Duty Alignment): High alignment ({duty_matches} duty overlaps) -> 25.0 points.")
        elif duty_matches == 1:
            v_align = 10.0
            diagnostics.append("Pillar C (Duty Alignment): Moderate alignment (1 duty overlap) -> 10.0 points.")
        else:
            v_align = 5.0
            diagnostics.append("Pillar C (Duty Alignment): General technical implementation -> 5.0 points.")

        return v_align
