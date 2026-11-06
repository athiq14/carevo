"""
Comprehensive unit tests for EntityExtractor and ResumeEntities.
Verifies skills categorization, work experience parsing, education degree extraction,
and project parsing on synthetic resume structures, including regression tests for URL handling,
refactored taxonomies, and ambiguity indicators.
"""
import pytest

from models.resume_entities import (
    EducationRecord,
    ExtractedSkill,
    ProjectRecord,
    ResumeEntities,
    SkillCategory,
    WorkExperience,
)
from models.resume_sections import ParsedResumeSections, ResumeSection, SectionType
from processors.entity_extractor import (
    EducationExtractor,
    EntityExtractor,
    ExperienceExtractor,
    ProjectExtractor,
    SkillsExtractor,
    extract_resume_entities,
)


@pytest.fixture
def extractor() -> EntityExtractor:
    return EntityExtractor()


def test_empty_sections(extractor: EntityExtractor):
    """Verify empty parsed resume sections return empty entity collections."""
    empty_parsed = ParsedResumeSections(
        original_text="",
        cleaned_text="",
        sections=[],
        detected_section_names=[],
        undetected_section_names=[],
    )
    entities = extractor.extract_entities(empty_parsed)
    assert isinstance(entities, ResumeEntities)
    assert len(entities.skills) == 0
    assert len(entities.experience) == 0
    assert len(entities.education) == 0
    assert len(entities.projects) == 0


def test_multiple_skills_and_categories():
    """Verify skill categorization across multiple domains including new categories."""
    skills_text = (
        "Python, JavaScript, React, PostgreSQL, Docker, AWS, PyTorch, Pandas, Apache Kafka, Git"
    )
    extracted = SkillsExtractor.extract_from_skills_section(skills_text)
    names = {s.name: s.category for s in extracted}

    assert names["Python"] == SkillCategory.PROGRAMMING_LANGUAGES
    assert names["React"] == SkillCategory.WEB_DEVELOPMENT
    assert names["PostgreSQL"] == SkillCategory.DATABASES
    assert names["AWS"] == SkillCategory.CLOUD_COMPUTING
    assert names["Docker"] == SkillCategory.DEVOPS_AND_INFRASTRUCTURE
    assert names["PyTorch"] == SkillCategory.MACHINE_LEARNING
    assert names["Pandas"] == SkillCategory.DATA_SCIENCE
    assert names["Apache Kafka"] == SkillCategory.DATA_ENGINEERING_AND_STREAMING
    assert names["Git"] == SkillCategory.TOOLS_AND_FRAMEWORKS


def test_grouped_skills_parsing():
    """Verify grouped category lines like 'Languages: ...' are parsed correctly."""
    grouped_text = (
        "Programming Languages: Python, Go, Rust\n"
        "Databases: PostgreSQL, Redis, MongoDB\n"
        "DevOps and Infrastructure: Docker, Kubernetes, CI/CD, Terraform\n"
        "Data Engineering & Streaming: Apache Kafka, Apache Spark, Airflow"
    )
    skills = SkillsExtractor.extract_from_skills_section(grouped_text)
    skill_map = {s.name: s.category for s in skills}

    assert "Python" in skill_map
    assert "Go" in skill_map
    assert "PostgreSQL" in skill_map
    assert "Redis" in skill_map
    assert "Docker" in skill_map
    assert "Kubernetes" in skill_map
    assert "CI/CD" in skill_map
    assert "Apache Kafka" in skill_map
    assert "Apache Spark" in skill_map
    assert "Airflow" in skill_map

    assert skill_map["Docker"] == SkillCategory.DEVOPS_AND_INFRASTRUCTURE
    assert skill_map["CI/CD"] == SkillCategory.DEVOPS_AND_INFRASTRUCTURE
    assert skill_map["Apache Kafka"] == SkillCategory.DATA_ENGINEERING_AND_STREAMING


def test_duplicate_skills_deduplication(extractor: EntityExtractor):
    """Verify that duplicate skill mentions in the skills section are deduplicated."""
    skills_text = "Python, python, PYTHON, Docker, docker, DOCKER"
    sec = ResumeSection(
        section_type=SectionType.TECHNICAL_SKILLS,
        original_heading="SKILLS",
        content=skills_text,
        start_line=1,
        end_line=1,
    )
    parsed = ParsedResumeSections(
        original_text=skills_text,
        cleaned_text=skills_text,
        sections=[sec],
        detected_section_names=["Technical Skills"],
        undetected_section_names=[],
    )
    entities = extractor.extract_entities(parsed)
    names = [s.name for s in entities.skills]
    assert names == ["Python", "Docker"]


def test_unlisted_skills_detected_in_experience(extractor: EntityExtractor):
    """Verify skills not listed in skills section are captured from experience with is_explicitly_listed=False."""
    skills_sec = ResumeSection(
        section_type=SectionType.TECHNICAL_SKILLS,
        original_heading="SKILLS",
        content="Python, SQL",
        start_line=1,
        end_line=1,
    )
    exp_sec = ResumeSection(
        section_type=SectionType.WORK_EXPERIENCE,
        original_heading="EXPERIENCE",
        content="Software Engineer | Tech Corp (2020 - Present)\n- Developed microservices in Go and deployed with Kubernetes.",
        start_line=3,
        end_line=5,
    )
    parsed = ParsedResumeSections(
        original_text="",
        cleaned_text="",
        sections=[skills_sec, exp_sec],
        detected_section_names=["Technical Skills", "Work Experience"],
        undetected_section_names=[],
    )
    entities = extractor.extract_entities(parsed)

    explicit_skills = {s.name: s.is_explicitly_listed for s in entities.skills}
    assert explicit_skills["Python"] is True
    assert explicit_skills["SQL"] is True
    assert explicit_skills["Go"] is False
    assert explicit_skills["Kubernetes"] is False


def test_multiple_job_entries_and_current_employment():
    """Verify extraction of multiple job entries with date indicators like 'Present'."""
    exp_text = (
        "Lead Backend Architect | Scale Systems (Jan 2021 - Present)\n"
        "- Spearheaded cloud infrastructure modernization.\n"
        "- Reduced latency by 35% using Redis caching.\n\n"
        "Senior Software Engineer at Alpha Labs | 2018 - 2021\n"
        "- Engineered real-time streaming pipeline with Apache Kafka and Python.\n"
        "- Maintained 99.99% availability."
    )
    jobs = ExperienceExtractor.extract(exp_text)
    assert len(jobs) == 2

    # Job 1
    assert "Lead Backend Architect" in (jobs[0].job_title or "")
    assert "Scale Systems" in (jobs[0].company or "")
    assert jobs[0].employment_start == "Jan 2021"
    assert jobs[0].employment_end == "Present"
    assert len(jobs[0].responsibilities) == 2
    assert "Redis" in jobs[0].technologies
    assert jobs[0].confidence == "CONFIRMED"
    assert jobs[0].is_ambiguous is False

    # Job 2
    assert "Senior Software Engineer" in (jobs[1].job_title or "")
    assert "Alpha Labs" in (jobs[1].company or "")
    assert jobs[1].employment_start == "2018"
    assert jobs[1].employment_end == "2021"
    assert "Apache Kafka" in jobs[1].technologies
    assert "Python" in jobs[1].technologies


def test_missing_employment_dates_handling():
    """Verify experience entry without dates populates None rather than failing."""
    exp_text = (
        "Principal Systems Consultant | Global Advisory\n"
        "- Led distributed architecture reviews and cloud risk assessments."
    )
    jobs = ExperienceExtractor.extract(exp_text)
    assert len(jobs) == 1
    assert jobs[0].employment_start is None
    assert jobs[0].employment_end is None
    assert jobs[0].job_title == "Principal Systems Consultant"
    assert jobs[0].company == "Global Advisory"
    assert jobs[0].confidence == "CONFIRMED"


def test_multiple_education_records_and_degree_formats():
    """Verify multiple degrees, majors, institutions, and date formats."""
    edu_text = (
        "Master of Science in Computer Science\n"
        "Stanford University (2018 - 2020)\n\n"
        "Bachelor of Science in Electrical Engineering\n"
        "University of California, Berkeley (2014 - 2018)\n\n"
        "Ph.D. in Artificial Intelligence\n"
        "MIT (2020 - 2024)"
    )
    records = EducationExtractor.extract(edu_text)
    assert len(records) == 3

    assert records[0].degree == "Master of Science"
    assert records[0].field_of_study == "Computer Science"
    assert records[0].institution == "Stanford University"
    assert records[0].start_date == "2018"
    assert records[0].end_date == "2020"
    assert records[0].confidence == "CONFIRMED"

    assert records[1].degree == "Bachelor of Science"
    assert records[1].field_of_study == "Electrical Engineering"
    assert records[1].institution == "University of California, Berkeley"

    assert "Ph.D" in (records[2].degree or "")
    assert records[2].field_of_study == "Artificial Intelligence"
    assert records[2].institution == "MIT"


def test_cgpa_and_percentage_extraction():
    """Verify CGPA, GPA, and percentage extraction across formats."""
    b1 = "B.S. in Computer Science\nCarnegie Mellon University (2016 - 2020)\nGPA: 3.92 / 4.0"
    b2 = "Bachelor of Technology in Information Technology\nDelhi Technological University\nCGPA: 8.9/10"
    b3 = "High School Diploma\nCity School\nGrade: 94.5%"

    r1 = EducationExtractor.extract(b1)[0]
    r2 = EducationExtractor.extract(b2)[0]
    r3 = EducationExtractor.extract(b3)[0]

    assert "3.92" in (r1.grade_or_cgpa or "")
    assert "8.9" in (r2.grade_or_cgpa or "")
    assert "94.5%" in (r3.grade_or_cgpa or "")


def test_multiple_projects_and_technologies():
    """Verify multiple projects with descriptions, extracted technologies, and URLs."""
    proj_text = (
        "- Distributed KV-Store: In-memory raft-consensus store built with Go and Docker.\n"
        "  Deployed on AWS with high resilience. https://github.com/example/kv-store\n\n"
        "- Real-Time Fraud Detection | Python, PyTorch, Apache Kafka\n"
        "  Engineered an anomaly detection microservice processing 50k transactions/sec."
    )
    projects = ProjectExtractor.extract(proj_text)
    assert len(projects) == 2

    # Project 1
    assert "Distributed KV-Store" in projects[0].project_name
    assert "Go" in projects[0].technologies
    assert "Docker" in projects[0].technologies
    assert "AWS" in projects[0].technologies
    assert projects[0].project_url == "https://github.com/example/kv-store"
    # Verify GitHub URL did NOT yield a Git skill!
    assert "Git" not in projects[0].technologies

    # Project 2
    assert "Real-Time Fraud Detection" in projects[1].project_name
    assert "PyTorch" in projects[1].technologies
    assert "Apache Kafka" in projects[1].technologies
    assert "Python" in projects[1].technologies
    assert projects[1].project_url is None


def test_malformed_and_ambiguous_content(extractor: EntityExtractor):
    """Verify resilience when sections contain unstructured or ambiguous text."""
    sec = ResumeSection(
        section_type=SectionType.WORK_EXPERIENCE,
        original_heading="WORK EXPERIENCE",
        content="General freelance consulting and varied tech engagements across multiple sectors.",
        start_line=1,
        end_line=2,
    )
    parsed = ParsedResumeSections(
        original_text="",
        cleaned_text="",
        sections=[sec],
        detected_section_names=["Work Experience"],
        undetected_section_names=[],
    )
    entities = extractor.extract_entities(parsed)

    # Should gracefully capture what is possible without crashing
    assert len(entities.experience) >= 1
    assert entities.experience[0].original_text != ""


# ---------------------------------------------------------------------------
# Targeted Regression Tests (Phase 2.3 Refinement)
# ---------------------------------------------------------------------------

def test_github_url_without_git_skill():
    """
    CRITICAL REGRESSION TEST:
    A GitHub URL must NEVER independently produce a 'Git' skill.
    """
    text = "Check out my distributed systems project at https://github.com/alex/distributed-cache"
    skills = SkillsExtractor.scan_mentioned_skills(text, source_section="Projects", already_extracted=set())

    extracted_names = [s.name for s in skills]
    assert "Git" not in extracted_names
    assert "GitHub" not in extracted_names


def test_git_explicitly_mentioned():
    """Verify 'Git' is extracted when explicitly mentioned in text."""
    text1 = "Version control: Git version control and branching workflows."
    text2 = "Used Git for version control across 15 engineering repositories."

    skills1 = SkillsExtractor.scan_mentioned_skills(text1, source_section="Work Experience", already_extracted=set())
    skills2 = SkillsExtractor.scan_mentioned_skills(text2, source_section="Work Experience", already_extracted=set())

    assert any(s.name == "Git" for s in skills1)
    assert any(s.name == "Git" for s in skills2)


def test_git_and_github_both_explicitly_mentioned():
    """Verify that when both Git and GitHub are explicitly mentioned, both are extracted."""
    skills_text = "Tools: Git, GitHub, GitLab, Jira"
    skills = SkillsExtractor.extract_from_skills_section(skills_text)
    names = {s.name for s in skills}

    assert "Git" in names
    assert "GitHub" in names
    assert "GitLab" in names
    assert "Jira" in names


def test_specific_skill_categorization_refinements():
    """
    Verify specific reclassifications:
    - Apache Kafka -> Data Engineering and Streaming
    - Docker, Kubernetes, CI/CD -> DevOps and Infrastructure
    """
    skills_text = "Apache Kafka, Docker, CI/CD, Kubernetes, Python"
    extracted = SkillsExtractor.extract_from_skills_section(skills_text)
    cat_map = {s.name: s.category for s in extracted}

    assert cat_map["Apache Kafka"] == SkillCategory.DATA_ENGINEERING_AND_STREAMING
    assert cat_map["Docker"] == SkillCategory.DEVOPS_AND_INFRASTRUCTURE
    assert cat_map["Kubernetes"] == SkillCategory.DEVOPS_AND_INFRASTRUCTURE
    assert cat_map["CI/CD"] == SkillCategory.DEVOPS_AND_INFRASTRUCTURE
    assert cat_map["Python"] == SkillCategory.PROGRAMMING_LANGUAGES


def test_missing_or_ambiguous_company_names():
    """Verify experience entries with ambiguous/missing company reflect AMBIGUOUS confidence."""
    ambiguous_text = (
        "Consultant / Contract Work\n"
        "- Delivered architecture optimizations and database tuning."
    )
    jobs = ExperienceExtractor.extract(ambiguous_text)
    assert len(jobs) == 1
    # Either title or company was unresolvable
    assert jobs[0].confidence == "AMBIGUOUS"
    assert jobs[0].is_ambiguous is True


def test_missing_education_dates_handling():
    """Verify education record handles missing dates without inventing them."""
    edu_text = (
        "Bachelor of Science in Computer Science\n"
        "University of Washington"
    )
    records = EducationExtractor.extract(edu_text)
    assert len(records) == 1
    assert records[0].start_date is None
    assert records[0].end_date is None
    assert records[0].institution == "University of Washington"
    assert records[0].degree == "Bachelor of Science"


def test_uncertain_project_technologies():
    """Verify project with uncertain or missing technologies is flagged as AMBIGUOUS."""
    proj_text = "Personal Portfolio: A simple website."
    records = ProjectExtractor.extract(proj_text)
    assert len(records) == 1
    assert len(records[0].technologies) == 0
    assert records[0].confidence == "AMBIGUOUS"
    assert records[0].is_ambiguous is True


def test_education_institution_on_separate_line_with_pipe_and_dates():
    """Regression test: verify institution on separate line with pipe and date range."""
    edu_text = (
        "Bachelor of Science in Software Engineering\n"
        "MIT | 2017 - 2021\n"
        "GPA: 3.9 / 4.0"
    )
    records = EducationExtractor.extract(edu_text)
    assert len(records) == 1
    rec = records[0]
    assert rec.degree == "Bachelor of Science"
    assert rec.field_of_study == "Software Engineering"
    assert rec.institution == "MIT"
    assert rec.start_date == "2017"
    assert rec.end_date == "2021"
    assert "3.9" in (rec.grade_or_cgpa or "")
    assert rec.confidence == "CONFIRMED"
    assert rec.is_ambiguous is False


def test_education_missing_institution_remains_none():
    """Verify that when no institution is present, it remains strictly None with AMBIGUOUS confidence."""
    edu_text = (
        "Bachelor of Science in Software Engineering\n"
        "2017 - 2021\n"
        "GPA: 3.9 / 4.0"
    )
    records = EducationExtractor.extract(edu_text)
    assert len(records) == 1
    rec = records[0]
    assert rec.degree == "Bachelor of Science"
    assert rec.field_of_study == "Software Engineering"
    assert rec.institution is None
    assert rec.start_date == "2017"
    assert rec.end_date == "2021"
    assert "3.9" in (rec.grade_or_cgpa or "")
    assert rec.confidence == "AMBIGUOUS"
    assert rec.is_ambiguous is True


def test_education_institution_variations_on_separate_lines():
    """Verify various separator formats (pipe, hyphen, acronyms) resolve cleanly without trailing delimiters."""
    # Format 1: University with trailing pipe stripped cleanly
    r1 = EducationExtractor.extract("Bachelor of Science\nStanford University | 2017 - 2021")[0]
    assert r1.institution == "Stanford University"
    assert r1.confidence == "CONFIRMED"

    # Format 2: Acronym with pipe and location
    r2 = EducationExtractor.extract("Master of Science in Robotics\nCMU | Pittsburgh, PA | 2018 - 2020")[0]
    assert r2.institution == "CMU"
    assert r2.confidence == "CONFIRMED"

    # Format 3: Acronym campus name
    r3 = EducationExtractor.extract("Bachelor of Technology\nIIT Madras | 2016 - 2020")[0]
    assert r3.institution == "IIT Madras"
    assert r3.confidence == "CONFIRMED"

