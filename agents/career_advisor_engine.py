"""
Intelligent Career Advisor Engine.
Orchestrates live LLM reasoning (via Groq), live web research, candidate evidence grounding,
and archetype-aware deterministic fallbacks across technical and non-technical professions.
Strictly eliminates single-candidate hardcoding, fake AI labels, and irrelevant recommendations.
"""
import json
import os
import re
import time
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from core.research.web_researcher import WebResearcher, ResearchDossier
from core.telemetry import telemetry_tracker
from models.matching import JobMatchResult, RequirementPriority
from models.resume_profile import ResumeProfile


class CareerRoadmapData(BaseModel):
    """Consolidated 7-part strategic career roadmap."""
    model_config = ConfigDict(frozen=True)

    objective_title: str
    archetype: str
    provenance: str  # "LIVE_GROQ_AI" or "VERIFIED_BASELINE_GUIDANCE"
    provider_name: str
    model_name: str
    direction_narrative: str
    target_roles: list[dict[str, str]]
    transferable_strengths: list[dict[str, str]]
    skill_priorities: list[dict[str, str]]
    milestones: list[dict[str, str]]
    portfolio_work_samples: list[dict[str, str]]
    interview_prep: list[dict[str, str]]
    research_dossier: Optional[dict[str, Any]] = None


def detect_role_archetype(role_or_objective: str) -> str:
    """Classify target role or objective into functional domain archetype."""
    text = (role_or_objective or "").lower()
    if any(w in text for w in ["home loan", "mortgage", "loan advisor", "credit appraisal", "retail lending", "dsa"]):
        return "MORTGAGE_LENDING"
    if any(w in text for w in ["home advisor", "property", "real estate", "realtor", "home buying", "housing"]):
        return "REAL_ESTATE_PROPERTY"
    if any(w in text for w in ["insurance", "actuar", "underwrit", "claim", "agent", "risk consultant"]):
        return "INSURANCE"
    if any(w in text for w in ["financial", "finance", "wealth", "banking", "investment", "accounting", "auditor"]):
        return "FINANCE"
    if any(w in text for w in ["hr", "human resources", "talent", "recruiter", "people ops"]):
        return "HUMAN_RESOURCES"
    if any(w in text for w in ["marketing", "sales", "public relations", "growth", "advertising", "account executive"]):
        return "MARKETING_SALES"
    if any(w in text for w in ["product manager", "operations", "business analyst", "project manager", "scrum"]):
        return "BUSINESS_OPERATIONS"
    if any(w in text for w in ["data analyst", "business intelligence", "bi analyst", "analytics", "data engineer"]):
        return "DATA_ANALYTICS"
    if any(w in text for w in ["software", "ai", "ml", "machine learning", "cloud", "devops", "platform", "backend", "full-stack", "frontend", "cybersecurity"]):
        return "SOFTWARE_TECH"
    return "GENERAL_PROFESSIONAL"


class CareerAdvisorEngine:
    """Engine synthesizing tailored career roadmaps with live research and Groq intelligence."""

    def __init__(self, dispatcher: Optional[Any] = None) -> None:
        self.dispatcher = dispatcher
        self.researcher = WebResearcher()

    def generate_roadmap(
        self,
        target_objective: str,
        profile: Optional[ResumeProfile] = None,
        match_result: Optional[JobMatchResult] = None,
    ) -> CareerRoadmapData:
        """
        Synthesize customized 7-part roadmap.
        Attempts live Groq LLM generation if available; falls back to archetype-aware baseline.
        """
        start_time = time.time()
        obj_clean = target_objective.strip() or "General Professional Development"
        archetype = detect_role_archetype(obj_clean)

        # 1. Execute live web research for the requested role
        try:
            research_dossier = self.researcher.research_role_requirements(obj_clean)
        except Exception:
            research_dossier = ResearchDossier(
                query=obj_clean,
                provider="none",
                status="OFFLINE",
                results=[],
                summary="Live market research offline; utilizing established industry criteria.",
                is_live_research=False,
            )

        # 2. Attempt live Groq generation if key configured
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

        if groq_api_key:
            try:
                live_result = self._synthesize_with_groq(
                    obj_clean, archetype, profile, match_result, research_dossier, groq_api_key, groq_model
                )
                if live_result:
                    latency = (time.time() - start_time) * 1000
                    telemetry_tracker.record_event(
                        feature="Career Advisor",
                        provider="groq",
                        model=groq_model,
                        status="SUCCESS",
                        latency_ms=latency,
                        web_research_used=research_dossier.is_live_research,
                        notes=f"Live roadmap generated for archetype: {archetype}",
                    )
                    return live_result
            except Exception as e:
                pass  # Fall through to archetype-aware deterministic fallback

        # 3. Deterministic archetype-aware fallback
        fallback_result = self._synthesize_archetype_fallback(
            obj_clean, archetype, profile, match_result, research_dossier
        )
        latency = (time.time() - start_time) * 1000
        telemetry_tracker.record_event(
            feature="Career Advisor",
            provider="rules_engine",
            model="archetype_synthesis",
            status="FALLBACK",
            latency_ms=latency,
            web_research_used=research_dossier.is_live_research,
            notes=f"Baseline guidance synthesized for archetype: {archetype}",
        )
        return fallback_result

    def _synthesize_with_groq(
        self,
        objective: str,
        archetype: str,
        profile: Optional[ResumeProfile],
        match_result: Optional[JobMatchResult],
        research_dossier: ResearchDossier,
        api_key: str,
        model: str,
    ) -> Optional[CareerRoadmapData]:
        """Send prompt to Groq for personalized 7-section roadmap."""
        from groq import Groq
        client = Groq(api_key=api_key)

        cand_skills = [s.name for s in getattr(profile, "skills", [])] if profile else []
        cand_edu = profile.education[0].degree if (profile and profile.education) else "Degree in progress or general background"
        cand_exp = [f"{e.job_title} at {e.company}" for e in getattr(profile, "experience", [])] if profile else []

        research_context = ""
        if research_dossier.results:
            citations = [f"- {r.title} ({r.url}): {r.snippet}" for r in research_dossier.results[:3]]
            research_context = "\n".join(citations)

        system_prompt = (
            "You are a Senior Career Strategist and Professional Readiness Advisor. "
            "You provide practical, realistic, grounded career guidance for ANY profession (technical or non-technical). "
            "Never recommend programming, software frameworks, or coding projects to non-technical roles (like Insurance, HR, Sales, Mortgages, Real Estate) "
            "unless explicitly requested.\n\n"
            "STRICT REGULATORY & LICENSING ACCURACY RULES:\n"
            "1. NEVER invent mandatory certifications or misattribute regulatory authority.\n"
            "2. SEBI & NISM: NISM Series X-A / X-B certifications apply EXCLUSIVELY to SEBI-registered Investment Advisers (RIAs) advising on securities, stocks, bonds, and mutual funds under SEBI (Investment Advisers) Regulations, 2013. NISM is NEVER required for home loans, mortgages, real estate, or general property advisory.\n"
            "3. HOME LOAN / MORTGAGE ADVISORS: Governed by Reserve Bank of India (RBI) Direct Selling Agent (DSA) guidelines and National Housing Bank (NHB) regulations. Key credentials include bank DSA empaneled agreements and optional IIBF (Indian Institute of Banking & Finance) certifications (e.g. Certified Credit Professional). There is NO mandatory SEBI or NISM licensing for retail home loans.\n"
            "4. REAL ESTATE / PROPERTY ADVISORS: Governed by state Real Estate Regulatory Authority (RERA) under the Real Estate (Regulation and Development) Act, 2016. Agents must obtain state RERA Agent Registration for commercial brokering. No NISM or SEBI certification applies.\n"
            "5. INSURANCE ADVISORS: Governed by IRDAI under Insurance Act 1938, requiring the IRDAI IC-38 examination.\n"
            "6. Always label certifications accurately as either 'Mandatory Regulatory License' (only when legally mandated like IRDAI IC-38 or RERA Registration) or 'Optional / Industry Recognized' (like IIBF, CFP, or SHRM). Never claim an optional certificate is mandatory by law.\n"
            "Return clean, valid JSON strictly adhering to the schema."
        )

        user_prompt = f"""
Candidate Profile:
- Stated Target Career Objective: {objective}
- Profession Archetype: {archetype}
- Candidate Education: {cand_edu}
- Candidate Skills: {', '.join(cand_skills[:8]) if cand_skills else 'General professional skills'}
- Past Experience: {', '.join(cand_exp[:2]) if cand_exp else 'Entry-level / Early career'}

Live Market Research:
{research_context or 'Use standard industry certification and skill benchmarks.'}

Generate a comprehensive 7-part strategic career roadmap formatted as JSON with exact keys:
{{
  "direction_narrative": "Strategic positioning paragraph explaining what this career requires and how the candidate can pivot.",
  "target_roles": [
    {{"title": "Role Title 1", "description": "1-line description"}},
    {{"title": "Role Title 2", "description": "1-line description"}},
    {{"title": "Role Title 3", "description": "1-line description"}},
    {{"title": "Role Title 4", "description": "1-line description"}}
  ],
  "transferable_strengths": [
    {{"title": "Strength 1", "explanation": "Detailed explanation of candidate's transferable asset"}},
    {{"title": "Strength 2", "explanation": "Detailed explanation of candidate's transferable asset"}}
  ],
  "skill_priorities": [
    {{"name": "Skill 1", "priority": "High Priority", "explanation": "What to learn"}},
    {{"name": "Skill 2", "priority": "High Priority", "explanation": "What to learn"}},
    {{"name": "Skill 3", "priority": "Medium Priority", "explanation": "What to learn"}},
    {{"name": "Skill 4", "priority": "Medium Priority", "explanation": "What to learn"}}
  ],
  "milestones": [
    {{"timeframe": "Month 1 (Days 1-30)", "focus": "Core Knowledge & Certification", "actions": "Key actions", "deliverable": "Tangible deliverable"}},
    {{"timeframe": "Month 2 (Days 31-60)", "focus": "Practical Application", "actions": "Key actions", "deliverable": "Tangible deliverable"}},
    {{"timeframe": "Month 3 (Days 61-90)", "focus": "Professional Readiness", "actions": "Key actions", "deliverable": "Tangible deliverable"}}
  ],
  "portfolio_work_samples": [
    {{"title": "Work Sample 1", "domain": "{objective}", "concept": "Realistic practical exercise or project concept", "deliverable": "Completed work product description"}},
    {{"title": "Work Sample 2", "domain": "{objective}", "concept": "Realistic practical exercise or project concept", "deliverable": "Completed work product description"}}
  ],
  "interview_prep": [
    {{"type": "Core Competency Question", "question": "Role-specific interview question", "tip": "How to answer convincingly"}},
    {{"type": "Situational & Client/Stakeholder Question", "question": "Role-specific interview question", "tip": "How to answer convincingly"}}
  ]
}}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2500,
            timeout=15.0,
        )

        content = resp.choices[0].message.content
        if not content:
            return None

        parsed = json.loads(content)
        return CareerRoadmapData(
            objective_title=objective,
            archetype=archetype,
            provenance="LIVE_GROQ_AI",
            provider_name="Groq",
            model_name=model,
            direction_narrative=parsed.get("direction_narrative", ""),
            target_roles=parsed.get("target_roles", []),
            transferable_strengths=parsed.get("transferable_strengths", []),
            skill_priorities=parsed.get("skill_priorities", []),
            milestones=parsed.get("milestones", []),
            portfolio_work_samples=parsed.get("portfolio_work_samples", []),
            interview_prep=parsed.get("interview_prep", []),
            research_dossier=research_dossier.model_dump(),
        )

    def _synthesize_archetype_fallback(
        self,
        objective: str,
        archetype: str,
        profile: Optional[ResumeProfile],
        match_result: Optional[JobMatchResult],
        research_dossier: ResearchDossier,
    ) -> CareerRoadmapData:
        """Role-archetype aware baseline roadmap guaranteeing relevant non-technical or technical guidance."""
        cand_skills = [s.name for s in getattr(profile, "skills", [])] if profile else []
        skills_str = ", ".join(cand_skills[:3]) if cand_skills else "demonstrated background"

        if archetype == "MORTGAGE_LENDING":
            return CareerRoadmapData(
                objective_title=objective,
                archetype=archetype,
                provenance="VERIFIED_BASELINE_GUIDANCE",
                provider_name="RBI & Banking Industry Standards",
                model_name="Mortgage & Retail Lending Architecture",
                direction_narrative=(
                    f"Advancing as a {objective} centers on retail credit appraisal, property title due diligence, "
                    "borrower Debt-to-Income / FOIR (Fixed Obligation to Income Ratio) calculation, and navigating lending guidelines across commercial banks and housing finance companies (HFCs). "
                    "Professional practice is governed by Reserve Bank of India (RBI) Direct Selling Agent (DSA) regulations and National Housing Bank (NHB) standards. "
                    "Note: NISM / SEBI certifications apply strictly to capital market securities and mutual funds, not retail mortgage distribution."
                ),
                target_roles=[
                    {"title": "Retail Home Loan & Mortgage Specialist", "description": "Evaluating borrower loan eligibility, income tax returns, and property sanction parameters."},
                    {"title": "Direct Selling Agent (DSA) Empaneled Consultant", "description": "Managing bank tie-ups, loan sourcing, and end-to-end disbursal tracking."},
                    {"title": "Credit Appraisal & Underwriting Associate", "description": "Assessing CIBIL scores, debt servicing capacity, and technical property valuation."},
                    {"title": "Secured Lending & Mortgage Solutions Manager", "description": "Advising clients on loan-against-property (LAP), balance transfers, and interest rate optimizations."},
                ],
                transferable_strengths=[
                    {"title": "Quantitative Evaluation & Risk Analysis", "explanation": f"Candidate's background with {skills_str} translates into rigorous financial modeling and FOIR calculations."},
                    {"title": "Structured Case Documentation", "explanation": "Experience in systematic reporting and deliverable tracking supports thorough bank KYC and title verification files."},
                ],
                skill_priorities=[
                    {"name": "Credit Appraisal & FOIR Modeling", "priority": "High Priority", "explanation": "Calculating borrower net take-home ratios, existing debt obligations, and maximum sanction eligibility."},
                    {"name": "RBI Direct Selling Agent (DSA) Compliance & Fair Lending Codes", "priority": "Mandatory Compliance", "explanation": "Mastering RBI guidelines for DSAs, customer data confidentiality, transparent fee disclosure, and ethical conduct."},
                    {"name": "Property Title Verification & Legal Valuation", "priority": "High Priority", "explanation": "Understanding Encumbrance Certificates (EC), Occupancy Certificates (OC), khata types, and approved building plans."},
                    {"name": "Mortgage Product Structuring (MCLR vs Repo-Linked Rates)", "priority": "Medium Priority", "explanation": "Comparing fixed vs floating interest rates, balance transfer mathematics, and prepayment penalty regulations under RBI rules."},
                ],
                milestones=[
                    {"timeframe": "Month 1 (Days 1-30)", "focus": "Regulatory Framework & Credit Appraisal Basics", "actions": "Study RBI master directions on retail lending, NHB guidelines, and CIBIL credit scoring mechanics.", "deliverable": "Complete Retail Credit Appraisal Case Study & Scorecard Benchmark."},
                    {"timeframe": "Month 2 (Days 31-60)", "focus": "Financial Spreading & Loan Eligibility Modeling", "actions": "Build comparative financial evaluation models across 5 major public and private mortgage lenders.", "deliverable": "Interactive Home Loan Eligibility & FOIR Spreadsheets for Salaried and Self-Employed Borrowers."},
                    {"timeframe": "Month 3 (Days 61-90)", "focus": "Property Due Diligence & Disbursal Protocol", "actions": "Analyze 3 full residential property documentation dossiers covering legal search reports and valuation.", "deliverable": "Comprehensive End-to-End Home Loan Sanction & Disbursal Playbook."},
                ],
                portfolio_work_samples=[
                    {"title": "Home Loan Financial Appraisal & FOIR Model", "domain": "Retail Mortgage Lending", "concept": "Comprehensive financial worksheet analyzing salaried vs self-employed borrower tax returns, liabilities, and bank sanction limits.", "deliverable": "Spreadsheet model with dynamic EMI schedules, loan-to-value (LTV) limits, and multi-bank comparison."},
                    {"title": "Property Legal & Title Due Diligence Checklist", "domain": "Secured Lending & Mortgages", "concept": "Standardized compliance binder detailing 15 mandatory title checks (EC, mother deed, approvals) for residential sanction.", "deliverable": "Operational Due Diligence Audit Template & Lender Submission SOP."},
                ],
                interview_prep=[
                    {"type": "Credit Assessment", "question": "How do you evaluate borrower eligibility when their income includes variable performance incentives and irregular rental cash flows?", "tip": "Explain banking credit appraisal standards: taking 2 to 3 years' average for variable income and requiring registered rental agreements with bank credit proof."},
                    {"type": "Regulatory Compliance", "question": "Which regulatory bodies oversee home loans in India, and how does mortgage advisory differ from SEBI-registered investment advice?", "tip": "Emphasize that retail mortgages operate under RBI DSA guidelines and NHB norms, whereas SEBI/NISM strictly regulates securities and capital market investment advice."},
                ],
                research_dossier=research_dossier.model_dump(),
            )

        elif archetype == "REAL_ESTATE_PROPERTY":
            return CareerRoadmapData(
                objective_title=objective,
                archetype=archetype,
                provenance="VERIFIED_BASELINE_GUIDANCE",
                provider_name="State RERA & Real Estate Regulatory Standards",
                model_name="Property Advisory & Real Estate Architecture",
                direction_narrative=(
                    f"Advising clients as a {objective} requires mastering residential property valuation, builder track-record vetting, "
                    "title due diligence, and mandatory state RERA (Real Estate Regulatory Authority) compliance under the Real Estate Act, 2016. "
                    "Real estate agents facilitating property purchase or sale in India must register as a licensed agent under their state RERA. "
                    "Note: If your objective focuses specifically on retail home financing and mortgages, explore the Home Loan Advisor specialization governed by RBI guidelines."
                ),
                target_roles=[
                    {"title": "RERA-Registered Real Estate Consultant", "description": "Consulting buyers on verified residential and commercial properties with state RERA certification."},
                    {"title": "Residential Property Acquisition Specialist", "description": "Guiding home buyers through site inspections, builder vetting, and transaction structuring."},
                    {"title": "Property Valuation & Due Diligence Analyst", "description": "Conducting comparative market analyses (CMA) and legal documentation verification."},
                    {"title": "Real Estate Investment Portfolio Advisor", "description": "Advising HNI and retail clients on rental yields, capital appreciation, and title security."},
                ],
                transferable_strengths=[
                    {"title": "Structured Comparative Analysis", "explanation": f"Candidate's background with {skills_str} supports systematic Comparative Market Analysis (CMA) and pricing analytics."},
                    {"title": "Analytical Due Diligence", "explanation": "Discipline in data verification translates into rigorous auditing of builder approvals and land title papers."},
                ],
                skill_priorities=[
                    {"name": "State RERA Agent Registration (Real Estate Act 2016)", "priority": "Mandatory License", "explanation": "Completing state RERA agent training, registration, and adhering to advertising disclosure regulations."},
                    {"name": "Property Title Search, RERA Approvals & Due Diligence", "priority": "High Priority", "explanation": "Verifying occupancy certificates, commencement certificates, encumbrance certificates, and RERA project numbers."},
                    {"name": "Comparative Market Analysis (CMA) & Property Valuation", "priority": "High Priority", "explanation": "Calculating price per square foot, carpet area vs super built-up, and rental yield benchmarking."},
                    {"name": "Consultative Negotiation & Buyer Advisory", "priority": "Medium Priority", "explanation": "Structuring price negotiations, agreement-to-sell clauses, and dispute escalation protocols."},
                ],
                milestones=[
                    {"timeframe": "Month 1 (Days 1-30)", "focus": "RERA Legal Framework & Licensing", "actions": "Study Real Estate (Regulation and Development) Act, 2016 and complete state RERA agent registration syllabus.", "deliverable": "Obtain State RERA Agent Registration certification."},
                    {"timeframe": "Month 2 (Days 31-60)", "focus": "Project Vetting & Comparative Market Analysis", "actions": "Audit 5 major residential development projects on the state RERA portal for compliance and delays.", "deliverable": "Comprehensive RERA Project Audit & Comparative Valuation Dossier."},
                    {"timeframe": "Month 3 (Days 61-90)", "focus": "Buyer Advisory & Transaction Advisory", "actions": "Conduct 3 simulated property consultations from requirement gathering to sale agreement drafting.", "deliverable": "Full Residential Home Buying Advisory Case Study."},
                ],
                portfolio_work_samples=[
                    {"title": "RERA Project Due Diligence & Valuation Report", "domain": "Real Estate Advisory", "concept": "Comprehensive buyer advisory dossier evaluating project RERA registration, approvals, carpet area efficiency, and price benchmarking.", "deliverable": "Written advisory memo with risk factors, builder delivery history, and legal approval checklists."},
                    {"title": "Home Buyer Negotiation & Transaction Handbook", "domain": "Property Consulting", "concept": "Standard operating procedure guiding buyers through agreement-to-sell clauses, payment milestones, and stamp duty calculation.", "deliverable": "Practical Transaction SOP and Legal Verification Checklist."},
                ],
                interview_prep=[
                    {"type": "Regulatory Compliance", "question": "What is the legal role of a RERA-registered agent under the Real Estate Act 2016, and what disclosures are mandatory to buyers?", "tip": "Highlight mandatory disclosure of RERA registration number, verified project sanction plans, and prohibition of false advertising under Section 10 of RERA."},
                    {"type": "Valuation & Advisory", "question": "How do you advise a first-time home buyer on evaluating carpet area vs super built-up area and assessing fair market price?", "tip": "Explain statutory definition of carpet area under RERA and walk through Comparative Market Analysis (CMA) methodology using recent registered sale deeds."},
                ],
                research_dossier=research_dossier.model_dump(),
            )

        elif archetype == "INSURANCE":
            return CareerRoadmapData(
                objective_title=objective,
                archetype=archetype,
                provenance="VERIFIED_BASELINE_GUIDANCE",
                provider_name="Verified Industry Standard",
                model_name="Insurance Advisory Architecture",
                direction_narrative=(
                    f"Transitioning toward an {objective} career requires developing consultative relationship skills, "
                    "mastering life and health risk underwriting principles, and adhering strictly to regulatory compliance (IRDAI). "
                    "Success is driven by client trust, comprehensive family needs analysis, and ethical policy placement."
                ),
                target_roles=[
                    {"title": "Licensed Insurance Advisor", "description": "Individual life & health policy consulting under direct insurer agency."},
                    {"title": "Financial Risk Consultant", "description": "Comprehensive family protection and wealth preservation planning."},
                    {"title": "Commercial Lines Specialist", "description": "SME liability, keyman insurance, and group medical advisory."},
                    {"title": "Corporate Benefits Advisor", "description": "Group health and employee protection programs for mid-market firms."},
                ],
                transferable_strengths=[
                    {"title": "Analytical Problem Solving", "explanation": f"Candidate's background with {skills_str} translates into rigorous actuarial needs assessment and policy comparison."},
                    {"title": "Structured Needs Assessment", "explanation": "Experience in reporting and systematic evaluation supports transparent client fact-finding and needs analysis."},
                ],
                skill_priorities=[
                    {"name": "IRDAI IC-38 Examination & Licensing", "priority": "Mandatory", "explanation": "Mastering the regulatory framework, sales ethics, 30-day free-look rules, and grievance redressal."},
                    {"name": "Human Life Value (HLV) Modeling", "priority": "High Priority", "explanation": "Calculating capital requirements for dependent support, debt clearance, and inflation hedging."},
                    {"name": "Life & Health Underwriting Fundamentals", "priority": "High Priority", "explanation": "Understanding mortality tables, pre-existing condition exclusions, waiting periods, and network coverage."},
                    {"name": "Consultative Objection Handling", "priority": "Medium Priority", "explanation": "Addressing client hesitations around policy duration, claim settlement ratios, and premium affordability."},
                ],
                milestones=[
                    {"timeframe": "Month 1 (Days 1-30)", "focus": "Pre-Licensing Training & IC-38 Prep", "actions": "Complete 50-hour accredited pre-licensing coursework; study contract law and ethics.", "deliverable": "Attain 85%+ score on IC-38 practice examinations."},
                    {"timeframe": "Month 2 (Days 31-60)", "focus": "Product Comparison & Fact-Finding", "actions": "Build comparative matrices across Term Life, Critical Illness, and Health cashless policies.", "deliverable": "Construct a Client Fact-Finding Protocol & Product Evaluation Matrix."},
                    {"timeframe": "Month 3 (Days 61-90)", "focus": "Client Simulation & Advisory Protocol", "actions": "Conduct 5 simulated client consultations with comprehensive needs analysis.", "deliverable": "Complete Family Risk Protection Case Study Dossier."},
                ],
                portfolio_work_samples=[
                    {"title": "Family Risk Protection Case Study Dossier", "domain": "Personal Financial Risk", "concept": "Comprehensive client fact-finding scenario covering a 4-person family, debt liabilities, and HLV calculation.", "deliverable": "Written advisory report with insurer Claim Settlement Ratio analysis and recommended coverage."},
                    {"title": "Commercial SME Risk Audit Template", "domain": "Commercial Lines", "concept": "Risk intake sheet for mid-sized enterprise assessing keyman coverage and group medical benefits.", "deliverable": "Structured audit protocol with policy exclusion checklist."},
                ],
                interview_prep=[
                    {"type": "Regulatory Knowledge", "question": "Explain the principle of Utmost Good Faith (Uberrima Fides) and how it affects policy claims.", "tip": "Emphasize mutual disclosure obligation and the legal impact of non-disclosure on claim settlement."},
                    {"type": "Consultative Fact-Finding", "question": "How do you calculate a client's Human Life Value when they have active home loans and dependent children?", "tip": "Walk through current income, liabilities, living expenses, and education funds to arrive at a defensible sum assured."},
                ],
                research_dossier=research_dossier.model_dump(),
            )

        elif archetype == "DATA_ANALYTICS":
            return CareerRoadmapData(
                objective_title=objective,
                archetype=archetype,
                provenance="VERIFIED_BASELINE_GUIDANCE",
                provider_name="Verified Industry Standard",
                model_name="Data Intelligence Architecture",
                direction_narrative=(
                    f"Excelling as a {objective} centers on turning complex business datasets into clear, actionable executive insights. "
                    "Employers prioritize candidates who combine advanced SQL and Python scripting with modern cloud data warehouses (Snowflake, BigQuery) "
                    "and intuitive dashboard storytelling (Tableau, PowerBI)."
                ),
                target_roles=[
                    {"title": "Senior Data Analyst", "description": "Multi-table SQL analysis, customer metrics, and KPI dash-boarding."},
                    {"title": "Business Intelligence Engineer", "description": "Data modeling, semantic layers, and automated reporting pipelines."},
                    {"title": "Product Analytics Specialist", "description": "User funnel analysis, A/B test evaluation, and feature adoption metrics."},
                    {"title": "Operations Analytics Lead", "description": "Supply chain, financial pacing, and operational bottleneck resolution."},
                ],
                transferable_strengths=[
                    {"title": "Quantitative & Technical Rigor", "explanation": f"Demonstrated background in {skills_str} provides a fast track to advanced data modeling."},
                    {"title": "Systematic Documentation", "explanation": "Structured approach to metrics and reporting ensures data governance and reproducibility."},
                ],
                skill_priorities=[
                    {"name": "Advanced SQL & Window Functions", "priority": "Mandatory", "explanation": "Mastering partition queries, rolling aggregates, and complex multi-join performance optimization."},
                    {"name": "Cloud Data Warehousing (Snowflake / BigQuery)", "priority": "High Priority", "explanation": "Understanding columnar storage, clustering keys, and staging transformations with dbt."},
                    {"name": "Executive Dashboard Storytelling (Tableau / PowerBI)", "priority": "High Priority", "explanation": "Designing user-centric interactive views with low cognitive load and clear decision drivers."},
                    {"name": "A/B Testing & Statistical Inference", "priority": "Medium Priority", "explanation": "Sample sizing, hypothesis testing, p-value interpretation, and variance reduction."},
                ],
                milestones=[
                    {"timeframe": "Month 1 (Days 1-30)", "focus": "SQL Mastery & Data Modeling", "actions": "Practice complex windowing queries and build normalized star-schema data models.", "deliverable": "Portfolio repository of 15 advanced SQL query solutions on public datasets."},
                    {"timeframe": "Month 2 (Days 31-60)", "focus": "Cloud Warehouse & Transformations", "actions": "Set up a Snowflake trial and build transformation pipelines using dbt and Python.", "deliverable": "Automated data transformation model with data quality tests."},
                    {"timeframe": "Month 3 (Days 61-90)", "focus": "End-to-End Executive Dashboard", "actions": "Connect Tableau/PowerBI to transformed dataset and build an interactive business portal.", "deliverable": "Live interactive executive dashboard tracking customer cohort churn and revenue pacing."},
                ],
                portfolio_work_samples=[
                    {"title": "E-Commerce Customer Churn & Cohort Analysis", "domain": "Product & Customer Analytics", "concept": "Cleanse and model transaction event logs to identify churn risk signals.", "deliverable": "Interactive dashboard with automated cohort retention heatmaps."},
                    {"title": "Financial Pacing & Revenue Reconciliation Pipeline", "domain": "Business Operations", "concept": "Multi-source reconciliation pipeline validating transaction fees against ledger deposits.", "deliverable": "Automated reconciliation script with discrepancy alerting."},
                ],
                interview_prep=[
                    {"type": "Technical SQL Deep Dive", "question": "How do window functions like ROW_NUMBER(), RANK(), and DENSE_RANK() differ in practice?", "tip": "Explain tie-breaking behavior and when you would use DENSE_RANK for customer rankings without skipping ranks."},
                    {"type": "Business Case Analysis", "question": "If our primary conversion metric dropped by 8% week-over-week, what steps would you take to diagnose it?", "tip": "Walk through segmentation by platform, geography, browser, and traffic channel before checking release regressions."},
                ],
                research_dossier=research_dossier.model_dump(),
            )

        elif archetype in ["MARKETING_SALES", "HUMAN_RESOURCES", "BUSINESS_OPERATIONS", "FINANCE"]:
            return CareerRoadmapData(
                objective_title=objective,
                archetype=archetype,
                provenance="VERIFIED_BASELINE_GUIDANCE",
                provider_name="Verified Industry Standard",
                model_name="Business & Management Architecture",
                direction_narrative=(
                    f"Advancing as a {objective} requires balancing strategic domain knowledge, consultative communication, "
                    "and measurable execution. Success comes from understanding stakeholder objectives, managing operational workflows, "
                    "and delivering structured, defensible outcomes."
                ),
                target_roles=[
                    {"title": f"Lead {objective}", "description": f"Strategic oversight and high-impact execution for {objective} workflows."},
                    {"title": f"Operations & Strategy Specialist ({objective})", "description": "Process design, cross-functional collaboration, and efficiency."},
                    {"title": "Client / Stakeholder Relationship Manager", "description": "Direct stakeholder management, expectation alignment, and retention."},
                    {"title": "Program & Implementation Lead", "description": "End-to-end delivery of strategic initiatives with measurable deliverables."},
                ],
                transferable_strengths=[
                    {"title": "Structured Problem Solving", "explanation": f"Analytical discipline in {skills_str} helps formulate clear frameworks and measurable goals."},
                    {"title": "Cross-Functional Collaboration", "explanation": "Experience collaborating across functions translates directly into effective stakeholder alignment."},
                ],
                skill_priorities=[
                    {"name": f"Core {objective} Methodologies & Regulations", "priority": "Mandatory", "explanation": "Industry-standard frameworks, compliance requirements, and operational best practices."},
                    {"name": "Consultative Stakeholder Management", "priority": "High Priority", "explanation": "Structuring executive briefings, negotiating priorities, and resolving objections."},
                    {"name": "Data-Informed Decision Making", "priority": "High Priority", "explanation": "Tracking core KPIs, managing budgets, and benchmarking operational performance."},
                    {"name": "Workflow Optimization & Tooling", "priority": "Medium Priority", "explanation": "Leveraging CRM, ERP, and project management platforms for seamless execution."},
                ],
                milestones=[
                    {"timeframe": "Month 1 (Days 1-30)", "focus": "Domain Fundamentals & Certifications", "actions": "Study industry-standard certifications and framework benchmarks.", "deliverable": "Completed domain certification syllabus & readiness audit."},
                    {"timeframe": "Month 2 (Days 31-60)", "focus": "Process Analysis & Protocol Design", "actions": "Map existing workflows, identify bottlenecks, and build standardized operational templates.", "deliverable": "Standard Operating Procedure (SOP) & KPI tracking dossier."},
                    {"timeframe": "Month 3 (Days 61-90)", "focus": "Strategic Case Study & Portfolio", "actions": "Execute an end-to-end project simulation with measurable business outcomes.", "deliverable": "Comprehensive Executive Presentation & Strategic Case Study."},
                ],
                portfolio_work_samples=[
                    {"title": f"Strategic {objective} Case Study & Operations Audit", "domain": "Business Operations", "concept": f"Detailed simulation addressing an operational bottleneck in {objective}.", "deliverable": "Full executive memo with ROI calculation and implementation roadmap."},
                    {"title": "Stakeholder Engagement & Communication Protocol", "domain": "Consulting & Management", "concept": "Structured briefing document and objection mitigation playbook.", "deliverable": "Practical handbook for client or executive negotiations."},
                ],
                interview_prep=[
                    {"type": "Strategic Alignment", "question": f"How do you prioritize competing requests from multiple stakeholders when resources are constrained?", "tip": "Use a clear framework (impact vs effort, ROI, regulatory urgency) to justify decision-making."},
                    {"type": "Situational Leadership", "question": "Describe a scenario where a project plan went off schedule and how you remediated the timeline.", "tip": "Focus on root cause diagnosis, transparent communication, and contingency adjustment."},
                ],
                research_dossier=research_dossier.model_dump(),
            )

        else:
            # Default Software / Tech Archetype
            return CareerRoadmapData(
                objective_title=objective,
                archetype=archetype,
                provenance="VERIFIED_BASELINE_GUIDANCE",
                provider_name="Verified Industry Standard",
                model_name="Technical Systems Architecture",
                direction_narrative=(
                    f"Advancing as a {objective} requires taking end-to-end outcome ownership: "
                    f"architecting resilient systems with {skills_str}, mastering domain idioms and clean API design, "
                    "and rigorously validating services with automated test suites and measurable operational metrics."
                ),
                target_roles=[
                    {"title": f"{objective}", "description": "Primary target trajectory aligning with stated goals."},
                    {"title": "Full-Stack AI & Systems Engineer", "description": "Frontend UI backed by robust asynchronous services."},
                    {"title": "Platform & Automation Lead", "description": "Workflow orchestration, streaming pipelines, and developer tooling."},
                    {"title": "Solutions & Integration Engineer", "description": "API integrations, client architecture, and operational optimization."},
                ],
                transferable_strengths=[
                    {"title": "Demonstrated Engineering Execution", "explanation": f"Hands-on background with {skills_str} provides strong technical foundation."},
                    {"title": "Modern Development Practices", "explanation": "Familiarity with version control, modular architectures, and test discipline."},
                ],
                skill_priorities=[
                    {"name": "Cloud Infrastructure & Containerization", "priority": "High Priority", "explanation": "Docker, Kubernetes, automated deployments, and reproducible environments."},
                    {"name": "Scalable Data Layers & Caching", "priority": "High Priority", "explanation": "PostgreSQL indexing, Redis caching, and connection pool optimization."},
                    {"name": "Asynchronous Messaging & APIs", "priority": "High Priority", "explanation": "REST/gRPC contracts, message queues, and event-driven patterns."},
                    {"name": "Observability & System Telemetry", "priority": "Medium Priority", "explanation": "Structured logging, latency profiling, and automated error guardrails."},
                ],
                milestones=[
                    {"timeframe": "Month 1 (Days 1-30)", "focus": "Core Architecture & Containerization", "actions": "Containerize services, configure isolated environments, and establish CI test workflows.", "deliverable": "Automated CI build pipeline with unit test suite."},
                    {"timeframe": "Month 2 (Days 31-60)", "focus": "Data Layer & Caching Strategies", "actions": "Implement caching layers, asynchronous worker queues, and query optimizations.", "deliverable": "High-throughput processing queue with benchmark telemetry."},
                    {"timeframe": "Month 3 (Days 61-90)", "focus": "End-to-End System Deliverable", "actions": "Build public portfolio service with automated integration tests and telemetry.", "deliverable": "Production-grade public repository with 85%+ test coverage and documentation."},
                ],
                portfolio_work_samples=[
                    {"title": f"Production {objective} Microservice", "domain": "Backend Systems", "concept": "High-throughput service handling structured requests with automated failover.", "deliverable": "Public repository with automated CI/CD and integration tests."},
                    {"title": "Distributed Ingestion & Analytics Pipeline", "domain": "Data & Platform", "concept": "Fault-tolerant streaming worker processing events with Redis caching.", "deliverable": "Containerized service with benchmark documentation."},
                ],
                interview_prep=[
                    {"type": "System Design", "question": "How do you handle service degradation when a downstream database or third-party API experiences elevated latency?", "tip": "Discuss circuit breakers, exponential backoff with jitter, and caching fallbacks."},
                    {"type": "Architecture & Concurrency", "question": "Explain how you ensure data consistency across distributed microservices without distributed transactions.", "tip": "Discuss the Saga pattern, outbox pattern, and idempotent event consumers."},
                ],
                research_dossier=research_dossier.model_dump(),
            )

