"""
Deterministic resume entity extraction engine.
Extracts structured skills, work experience, education, and projects from parsed resume sections
with explicit evidence tracking and ambiguity classification.
"""
import re
from typing import Optional

from models.resume_entities import (
    EducationRecord,
    ExtractedSkill,
    ProjectRecord,
    ResumeEntities,
    SkillCategory,
    WorkExperience,
)
from models.resume_sections import ParsedResumeSections, SectionType


# Regex to sanitize and exclude web URLs from skill extraction matching
URL_EXCLUSION_PATTERN = re.compile(
    r'https?://[^\s\)]+|www\.[^\s\)]+|github\.com/[^\s\)]+|gitlab\.com/[^\s\)]+|linkedin\.com/[^\s\)]+',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Technical Skills Taxonomy and Category Mapping
# ---------------------------------------------------------------------------

SKILL_TAXONOMY: dict[str, tuple[SkillCategory, set[str]]] = {
    # Programming Languages
    "Python": (SkillCategory.PROGRAMMING_LANGUAGES, {"python", "python3", "py"}),
    "JavaScript": (SkillCategory.PROGRAMMING_LANGUAGES, {"javascript", "js", "ecmascript"}),
    "TypeScript": (SkillCategory.PROGRAMMING_LANGUAGES, {"typescript", "ts"}),
    "Java": (SkillCategory.PROGRAMMING_LANGUAGES, {"java", "core java"}),
    "C++": (SkillCategory.PROGRAMMING_LANGUAGES, {"c++", "cpp"}),
    "C#": (SkillCategory.PROGRAMMING_LANGUAGES, {"c#", "csharp", "c sharp"}),
    "C": (SkillCategory.PROGRAMMING_LANGUAGES, {"c language"}),
    "Go": (SkillCategory.PROGRAMMING_LANGUAGES, {"go", "golang"}),
    "Rust": (SkillCategory.PROGRAMMING_LANGUAGES, {"rust"}),
    "Ruby": (SkillCategory.PROGRAMMING_LANGUAGES, {"ruby"}),
    "PHP": (SkillCategory.PROGRAMMING_LANGUAGES, {"php"}),
    "Swift": (SkillCategory.PROGRAMMING_LANGUAGES, {"swift"}),
    "Kotlin": (SkillCategory.PROGRAMMING_LANGUAGES, {"kotlin"}),
    "Scala": (SkillCategory.PROGRAMMING_LANGUAGES, {"scala"}),
    "R": (SkillCategory.PROGRAMMING_LANGUAGES, {"r language"}),
    "SQL": (SkillCategory.PROGRAMMING_LANGUAGES, {"sql", "ansi sql", "pl/sql", "t-sql"}),
    "Shell": (SkillCategory.PROGRAMMING_LANGUAGES, {"bash", "shell", "powershell", "zsh"}),
    "HTML/CSS": (SkillCategory.PROGRAMMING_LANGUAGES, {"html", "html5", "css", "css3"}),
    "Dart": (SkillCategory.PROGRAMMING_LANGUAGES, {"dart"}),
    "Perl": (SkillCategory.PROGRAMMING_LANGUAGES, {"perl"}),

    # Web Development
    "React": (SkillCategory.WEB_DEVELOPMENT, {"react", "react.js", "reactjs"}),
    "Next.js": (SkillCategory.WEB_DEVELOPMENT, {"next.js", "nextjs", "next"}),
    "Node.js": (SkillCategory.WEB_DEVELOPMENT, {"node.js", "nodejs", "node"}),
    "Express": (SkillCategory.WEB_DEVELOPMENT, {"express", "express.js", "expressjs"}),
    "Angular": (SkillCategory.WEB_DEVELOPMENT, {"angular", "angularjs"}),
    "Vue.js": (SkillCategory.WEB_DEVELOPMENT, {"vue", "vue.js", "vuejs"}),
    "Django": (SkillCategory.WEB_DEVELOPMENT, {"django"}),
    "Flask": (SkillCategory.WEB_DEVELOPMENT, {"flask"}),
    "FastAPI": (SkillCategory.WEB_DEVELOPMENT, {"fastapi"}),
    "Spring Boot": (SkillCategory.WEB_DEVELOPMENT, {"spring boot", "springboot", "spring"}),
    "ASP.NET": (SkillCategory.WEB_DEVELOPMENT, {"asp.net", "dotnet", ".net", ".net core"}),
    "GraphQL": (SkillCategory.WEB_DEVELOPMENT, {"graphql"}),
    "REST APIs": (SkillCategory.WEB_DEVELOPMENT, {"rest api", "rest apis", "restful", "rest"}),
    "API Integration": (SkillCategory.WEB_DEVELOPMENT, {"api integration", "api integrations", "apis", "api design", "rest api integrations"}),
    "gRPC": (SkillCategory.WEB_DEVELOPMENT, {"grpc"}),
    "Tailwind CSS": (SkillCategory.WEB_DEVELOPMENT, {"tailwind", "tailwind css", "tailwindcss"}),
    "Bootstrap": (SkillCategory.WEB_DEVELOPMENT, {"bootstrap"}),

    # Databases
    "PostgreSQL": (SkillCategory.DATABASES, {"postgresql", "postgres", "psql"}),
    "MySQL": (SkillCategory.DATABASES, {"mysql"}),
    "MongoDB": (SkillCategory.DATABASES, {"mongodb", "mongo"}),
    "Redis": (SkillCategory.DATABASES, {"redis"}),
    "SQLite": (SkillCategory.DATABASES, {"sqlite", "sqlite3"}),
    "Oracle Database": (SkillCategory.DATABASES, {"oracle", "oracle db", "oracle database"}),
    "Microsoft SQL Server": (SkillCategory.DATABASES, {"mssql", "sql server", "microsoft sql server"}),
    "Cassandra": (SkillCategory.DATABASES, {"cassandra", "apache cassandra"}),
    "DynamoDB": (SkillCategory.DATABASES, {"dynamodb", "aws dynamodb"}),
    "Elasticsearch": (SkillCategory.DATABASES, {"elasticsearch", "elastic search", "opensearch"}),
    "Snowflake": (SkillCategory.DATABASES, {"snowflake"}),
    "Neo4j": (SkillCategory.DATABASES, {"neo4j"}),

    # Cloud Computing
    "AWS": (SkillCategory.CLOUD_COMPUTING, {"aws", "amazon web services", "ec2", "s3", "lambda"}),
    "Google Cloud Platform": (SkillCategory.CLOUD_COMPUTING, {"gcp", "google cloud", "google cloud platform"}),
    "Microsoft Azure": (SkillCategory.CLOUD_COMPUTING, {"azure", "microsoft azure"}),
    "Cloudflare": (SkillCategory.CLOUD_COMPUTING, {"cloudflare"}),
    "Serverless": (SkillCategory.CLOUD_COMPUTING, {"serverless"}),

    # DevOps and Infrastructure
    "Docker": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"docker"}),
    "Kubernetes": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"kubernetes", "k8s"}),
    "CI/CD": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"ci/cd", "ci-cd", "continuous integration", "continuous deployment"}),
    "Terraform": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"terraform"}),
    "Jenkins": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"jenkins"}),
    "GitHub Actions": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"github actions"}),
    "GitLab CI": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"gitlab ci", "gitlab-ci"}),
    "Ansible": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"ansible"}),
    "Prometheus": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"prometheus"}),
    "Grafana": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"grafana"}),
    "Helm": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"helm"}),
    "ArgoCD": (SkillCategory.DEVOPS_AND_INFRASTRUCTURE, {"argocd", "argo cd"}),

    # Machine Learning
    "PyTorch": (SkillCategory.MACHINE_LEARNING, {"pytorch"}),
    "TensorFlow": (SkillCategory.MACHINE_LEARNING, {"tensorflow", "tf"}),
    "Keras": (SkillCategory.MACHINE_LEARNING, {"keras"}),
    "Scikit-Learn": (SkillCategory.MACHINE_LEARNING, {"scikit-learn", "sklearn"}),
    "Hugging Face": (SkillCategory.MACHINE_LEARNING, {"hugging face", "huggingface", "transformers"}),
    "Computer Vision": (SkillCategory.MACHINE_LEARNING, {"computer vision", "opencv"}),
    "NLP": (SkillCategory.MACHINE_LEARNING, {"natural language processing", "nlp", "spacy", "nltk"}),
    "LLMs": (SkillCategory.MACHINE_LEARNING, {"llms", "large language models", "llm"}),
    "AI Applications": (SkillCategory.MACHINE_LEARNING, {"ai applications", "ai/llm applications", "llm applications", "ai solutions", "ai-first"}),
    "LangChain": (SkillCategory.MACHINE_LEARNING, {"langchain"}),
    "LangGraph": (SkillCategory.MACHINE_LEARNING, {"langgraph"}),
    "LlamaIndex": (SkillCategory.MACHINE_LEARNING, {"llamaindex"}),
    "XGBoost": (SkillCategory.MACHINE_LEARNING, {"xgboost"}),

    # Data Science
    "Pandas": (SkillCategory.DATA_SCIENCE, {"pandas"}),
    "NumPy": (SkillCategory.DATA_SCIENCE, {"numpy"}),
    "SciPy": (SkillCategory.DATA_SCIENCE, {"scipy"}),
    "Matplotlib": (SkillCategory.DATA_SCIENCE, {"matplotlib"}),
    "Seaborn": (SkillCategory.DATA_SCIENCE, {"seaborn"}),
    "Tableau": (SkillCategory.DATA_SCIENCE, {"tableau"}),
    "Power BI": (SkillCategory.DATA_SCIENCE, {"power bi", "powerbi"}),

    # Data Engineering and Streaming
    "Apache Kafka": (SkillCategory.DATA_ENGINEERING_AND_STREAMING, {"kafka", "apache kafka"}),
    "Apache Spark": (SkillCategory.DATA_ENGINEERING_AND_STREAMING, {"spark", "apache spark", "pyspark"}),
    "Airflow": (SkillCategory.DATA_ENGINEERING_AND_STREAMING, {"airflow", "apache airflow"}),
    "Apache Flink": (SkillCategory.DATA_ENGINEERING_AND_STREAMING, {"flink", "apache flink"}),
    "Hadoop": (SkillCategory.DATA_ENGINEERING_AND_STREAMING, {"hadoop", "apache hadoop"}),

    # Tools and Frameworks
    "Git": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"git", "git vcs", "git cli"}),
    "GitHub": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"github"}),
    "GitLab": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"gitlab"}),
    "Linux": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"linux", "unix", "ubuntu", "debian"}),
    "Jira": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"jira"}),
    "Streamlit": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"streamlit"}),
    "Pytest": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"pytest"}),
    "Postman": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"postman"}),
    "Celery": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"celery"}),
    "Automation": (SkillCategory.TOOLS_AND_FRAMEWORKS, {"automation", "workflow automation", "process automation", "automated pipelines"}),
}

# Inverted lookup for case-insensitive exact matching
_SKILL_LOOKUP: dict[str, tuple[str, SkillCategory]] = {}
for canonical_name, (cat, aliases) in SKILL_TAXONOMY.items():
    _SKILL_LOOKUP[canonical_name.lower()] = (canonical_name, cat)
    for alias in aliases:
        _SKILL_LOOKUP[alias.lower()] = (canonical_name, cat)

# Category label heuristic for grouped skill lines
_CATEGORY_HEADER_MAP: dict[str, SkillCategory] = {
    "programming": SkillCategory.PROGRAMMING_LANGUAGES,
    "programming languages": SkillCategory.PROGRAMMING_LANGUAGES,
    "languages": SkillCategory.PROGRAMMING_LANGUAGES,
    "web development": SkillCategory.WEB_DEVELOPMENT,
    "frontend": SkillCategory.WEB_DEVELOPMENT,
    "backend": SkillCategory.WEB_DEVELOPMENT,
    "web technologies": SkillCategory.WEB_DEVELOPMENT,
    "frameworks": SkillCategory.TOOLS_AND_FRAMEWORKS,
    "tools": SkillCategory.TOOLS_AND_FRAMEWORKS,
    "tools and frameworks": SkillCategory.TOOLS_AND_FRAMEWORKS,
    "tools & frameworks": SkillCategory.TOOLS_AND_FRAMEWORKS,
    "databases": SkillCategory.DATABASES,
    "database": SkillCategory.DATABASES,
    "storage": SkillCategory.DATABASES,
    "storage & databases": SkillCategory.DATABASES,
    "storage and databases": SkillCategory.DATABASES,
    "cloud": SkillCategory.CLOUD_COMPUTING,
    "cloud computing": SkillCategory.CLOUD_COMPUTING,
    "cloud & infrastructure": SkillCategory.CLOUD_COMPUTING,
    "cloud and infrastructure": SkillCategory.CLOUD_COMPUTING,
    "devops": SkillCategory.DEVOPS_AND_INFRASTRUCTURE,
    "devops and infrastructure": SkillCategory.DEVOPS_AND_INFRASTRUCTURE,
    "devops & infrastructure": SkillCategory.DEVOPS_AND_INFRASTRUCTURE,
    "infrastructure": SkillCategory.DEVOPS_AND_INFRASTRUCTURE,
    "machine learning": SkillCategory.MACHINE_LEARNING,
    "artificial intelligence": SkillCategory.MACHINE_LEARNING,
    "ai / ml": SkillCategory.MACHINE_LEARNING,
    "data science": SkillCategory.DATA_SCIENCE,
    "data engineering": SkillCategory.DATA_ENGINEERING_AND_STREAMING,
    "data engineering and streaming": SkillCategory.DATA_ENGINEERING_AND_STREAMING,
    "data engineering & streaming": SkillCategory.DATA_ENGINEERING_AND_STREAMING,
    "streaming": SkillCategory.DATA_ENGINEERING_AND_STREAMING,
    "analytics": SkillCategory.DATA_SCIENCE,
    "automation": SkillCategory.TOOLS_AND_FRAMEWORKS,
    "api": SkillCategory.WEB_DEVELOPMENT,
    "apis": SkillCategory.WEB_DEVELOPMENT,
}


# ---------------------------------------------------------------------------
# Skills Extractor
# ---------------------------------------------------------------------------

class SkillsExtractor:
    """Extracts, normalizes, and categorizes technical skills with evidence tracking."""

    @classmethod
    def extract_from_skills_section(cls, text: str) -> list[ExtractedSkill]:
        """Extracts explicitly listed skills from the Technical Skills section."""
        if not text:
            return []

        extracted_skills: dict[str, ExtractedSkill] = {}
        lines = text.splitlines()

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            # Strip leading bullet points
            clean_line = re.sub(r'^(?:[\-\*\•\●\▪\–\—]\s*)', '', clean_line).strip()

            # Check for grouped category format: e.g. "Programming: Python, Java, SQL"
            line_category: Optional[SkillCategory] = None
            content_part = clean_line

            if ":" in clean_line:
                prefix, remainder = clean_line.split(":", 1)
                prefix_clean = prefix.strip().lower()
                if prefix_clean in _CATEGORY_HEADER_MAP:
                    line_category = _CATEGORY_HEADER_MAP[prefix_clean]
                    content_part = remainder
                elif len(prefix.split()) <= 4:
                    content_part = remainder
            elif " - " in clean_line:
                prefix, remainder = clean_line.split(" - ", 1)
                prefix_clean = prefix.strip().lower()
                if prefix_clean in _CATEGORY_HEADER_MAP:
                    line_category = _CATEGORY_HEADER_MAP[prefix_clean]
                    content_part = remainder

            # Sanitize URLs out of content part so URLs never yield skills
            sanitized_content = URL_EXCLUSION_PATTERN.sub(" ", content_part)

            # Split tokens by comma, pipe, bullet, or semicolon
            raw_tokens = re.split(r'[,\|;•·\t]+', sanitized_content)

            for token in raw_tokens:
                token_clean = token.strip()
                if not token_clean or len(token_clean) < 2 or len(token_clean) > 40:
                    continue

                lookup_key = token_clean.lower()
                if lookup_key in _SKILL_LOOKUP:
                    canonical_name, detected_cat = _SKILL_LOOKUP[lookup_key]
                    # Specific canonical category from taxonomy takes precedence over broad line headers
                    if detected_cat and detected_cat != SkillCategory.OTHER_TECHNICAL_SKILLS:
                        cat = detected_cat
                    else:
                        cat = line_category if line_category is not None else detected_cat

                    if canonical_name not in extracted_skills:
                        extracted_skills[canonical_name] = ExtractedSkill(
                            name=canonical_name,
                            original_mention=token_clean,
                            category=cat,
                            source_section="Technical Skills",
                            is_explicitly_listed=True,
                            raw_evidence=line.strip(),
                            confidence="CONFIRMED",
                            is_ambiguous=False,
                        )
                elif line_category is not None:
                    # Token was explicitly listed under a known category heading (e.g. "Tools: CustomTool")
                    if token_clean not in extracted_skills and not any(c in token_clean for c in ".,;:"):
                        extracted_skills[token_clean] = ExtractedSkill(
                            name=token_clean,
                            original_mention=token_clean,
                            category=line_category,
                            source_section="Technical Skills",
                            is_explicitly_listed=True,
                            raw_evidence=line.strip(),
                            confidence="AMBIGUOUS",
                            is_ambiguous=True,
                        )

        return list(extracted_skills.values())

    @classmethod
    def scan_mentioned_skills(
        cls,
        text: str,
        source_section: str,
        already_extracted: set[str],
    ) -> list[ExtractedSkill]:
        """
        Scans arbitrary section text for technical skills from the taxonomy.
        Explicitly excludes URLs to prevent false positives (e.g. github.com -> Git).
        """
        if not text:
            return []

        # Sanitize URLs out of text prior to skill matching
        text_sanitized = URL_EXCLUSION_PATTERN.sub(" ", text)

        results: list[ExtractedSkill] = []
        seen_in_section: set[str] = set()

        for lookup_key, (canonical_name, cat) in _SKILL_LOOKUP.items():
            if canonical_name in already_extracted or canonical_name in seen_in_section:
                continue

            # Word boundary matching with special char escapes (e.g. C++, C#)
            pattern = rf'(?<![a-zA-Z0-9]){re.escape(lookup_key)}(?![a-zA-Z0-9])'
            match = re.search(pattern, text_sanitized, re.IGNORECASE)
            if match:
                matched_text = match.group(0)
                seen_in_section.add(canonical_name)

                # Flag ambiguous short tokens or generic words (e.g., 'go' as an English verb)
                is_ambiguous_mention = False
                if lookup_key.lower() == "go":
                    # Check if 'Go' is capitalized or associated with programming/lang context
                    if not re.search(r'\b(?:golang|go\s+language|programming|developer|written\s+in\s+go)\b', text_sanitized, re.IGNORECASE) and matched_text != "Go":
                        is_ambiguous_mention = True

                results.append(
                    ExtractedSkill(
                        name=canonical_name,
                        original_mention=matched_text,
                        category=cat,
                        source_section=source_section,
                        is_explicitly_listed=False,
                        raw_evidence=matched_text,
                        confidence="AMBIGUOUS" if is_ambiguous_mention else "CONFIRMED",
                        is_ambiguous=is_ambiguous_mention,
                    )
                )

        return results


# ---------------------------------------------------------------------------
# Work Experience Extractor
# ---------------------------------------------------------------------------

class ExperienceExtractor:
    """Extracts structured employment and internship history records with ambiguity tracking."""

    _DATE_RANGE_REGEX = re.compile(
        r'(?P<start>(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*\.?\s+)?'
        r'(?:19|20)\d{2})\s*(?:[-–—to]+)\s*'
        r'(?P<end>(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*\.?\s+)?'
        r'(?:19|20)\d{2}|Present|Current|Now|Ongoing)',
        re.IGNORECASE,
    )

    _SINGLE_DATE_REGEX = re.compile(
        r'\b(?:19|20)\d{2}\b',
    )

    _JOB_TITLE_KEYWORDS = {
        "engineer", "architect", "developer", "lead", "manager", "intern",
        "scientist", "specialist", "consultant", "director", "administrator",
        "analyst", "designer", "officer", "coordinator", "fellow", "programmer",
        "technician", "supervisor", "head", "vp", "associate"
    }

    @classmethod
    def _is_job_header_line(cls, line: str) -> bool:
        clean = line.strip()
        if not clean or clean.startswith(("-", "*", "•", "●")):
            return False

        has_date = bool(cls._DATE_RANGE_REGEX.search(clean))
        lower_line = clean.lower()
        has_title = any(kw in lower_line for kw in cls._JOB_TITLE_KEYWORDS)
        has_separator = any(sep in clean for sep in ["|", " - ", " at ", " @ ", " – "])

        return (has_date and has_title) or (has_date and has_separator) or (has_title and has_separator)

    @classmethod
    def _parse_job_header(cls, header_line: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parses job_title, company, employment_start, employment_end, employment_type."""
        start_date: Optional[str] = None
        end_date: Optional[str] = None
        emp_type: Optional[str] = None

        # Check employment arrangement
        for t in ["Full-time", "Part-time", "Contract", "Internship", "Remote"]:
            if re.search(rf'\b{t}\b', header_line, re.IGNORECASE):
                emp_type = t
                break

        # Extract dates
        date_match = cls._DATE_RANGE_REGEX.search(header_line)
        cleaned_header = header_line
        if date_match:
            start_date = date_match.group("start").strip()
            end_val = date_match.group("end").strip()
            end_date = "Present" if end_val.lower() in ("present", "current", "now", "ongoing") else end_val
            # Remove date portion for title/company parsing
            cleaned_header = header_line[:date_match.start()] + header_line[date_match.end():]
        else:
            single_dates = cls._SINGLE_DATE_REGEX.findall(header_line)
            if len(single_dates) >= 2:
                start_date = single_dates[0]
                end_date = single_dates[1]
            elif len(single_dates) == 1:
                end_date = single_dates[0]

        # Clean remaining title/company text
        cleaned_header = re.sub(r'[\(\)\[\]]', ' ', cleaned_header)
        cleaned_header = re.sub(r'[ \t]+', ' ', cleaned_header).strip().strip("|-–—,")

        job_title: Optional[str] = None
        company: Optional[str] = None

        parts = []
        if "|" in cleaned_header:
            parts = [p.strip() for p in cleaned_header.split("|") if p.strip()]
        elif " at " in cleaned_header.lower():
            idx = cleaned_header.lower().find(" at ")
            parts = [cleaned_header[:idx].strip(), cleaned_header[idx + 4:].strip()]
        elif " @ " in cleaned_header:
            parts = [p.strip() for p in cleaned_header.split(" @ ") if p.strip()]
        elif " – " in cleaned_header:
            parts = [p.strip() for p in cleaned_header.split(" – ") if p.strip()]
        elif " - " in cleaned_header:
            parts = [p.strip() for p in cleaned_header.split(" - ") if p.strip()]
        elif "," in cleaned_header:
            parts = [p.strip() for p in cleaned_header.split(",", 1) if p.strip()]

        if len(parts) >= 2:
            p1, p2 = parts[0], parts[1]
            p1_has_title = any(kw in p1.lower() for kw in cls._JOB_TITLE_KEYWORDS)
            p2_has_title = any(kw in p2.lower() for kw in cls._JOB_TITLE_KEYWORDS)

            if p1_has_title and not p2_has_title:
                job_title, company = p1, p2
            elif p2_has_title and not p1_has_title:
                job_title, company = p2, p1
            else:
                job_title, company = p1, p2
        elif len(parts) == 1:
            if any(kw in parts[0].lower() for kw in cls._JOB_TITLE_KEYWORDS):
                job_title = parts[0]
            else:
                company = parts[0]
        else:
            if cleaned_header:
                job_title = cleaned_header

        return job_title, company, start_date, end_date, emp_type

    @classmethod
    def extract(cls, text: str) -> list[WorkExperience]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        jobs: list[WorkExperience] = []

        current_header: Optional[str] = None
        current_lines: list[str] = []

        def commit_job():
            nonlocal current_header, current_lines
            if not current_header and not current_lines:
                return

            full_block = "\n".join(([current_header] if current_header else []) + current_lines).strip()
            title, comp, s_date, e_date, e_type = cls._parse_job_header(current_header or "")

            responsibilities: list[str] = []
            for ln in current_lines:
                cl = ln.strip()
                if cl.startswith(("-", "*", "•", "●")):
                    responsibilities.append(re.sub(r'^(?:[\-\*\•\●]\s*)', '', cl).strip())
                elif cl:
                    responsibilities.append(cl)

            # Exclude URLs from tech detection
            block_sanitized = URL_EXCLUSION_PATTERN.sub(" ", full_block)

            # Detect technologies mentioned in this role
            techs: list[str] = []
            for key, (canonical, _) in _SKILL_LOOKUP.items():
                if canonical not in techs:
                    pattern = rf'(?<![a-zA-Z0-9]){re.escape(key)}(?![a-zA-Z0-9])'
                    if re.search(pattern, block_sanitized, re.IGNORECASE):
                        techs.append(canonical)

            # Mark ambiguous if company or job title could not be confirmed
            is_ambiguous = (title is None or comp is None)
            confidence = "AMBIGUOUS" if is_ambiguous else "CONFIRMED"

            jobs.append(
                WorkExperience(
                    job_title=title,
                    company=comp,
                    employment_start=s_date,
                    employment_end=e_date,
                    employment_type=e_type,
                    responsibilities=responsibilities,
                    technologies=techs,
                    original_text=full_block,
                    confidence=confidence,
                    is_ambiguous=is_ambiguous,
                )
            )

        for line in lines:
            if cls._is_job_header_line(line):
                commit_job()
                current_header = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        commit_job()
        return jobs


# ---------------------------------------------------------------------------
# Education Extractor
# ---------------------------------------------------------------------------

class EducationExtractor:
    """Extracts degrees, majors, institutions, dates, and GPA with uncertainty tracking."""

    _DEGREE_PATTERN = re.compile(
        r'\b(Master(?:\s+of\s+(?:Science|Arts|Engineering|Technology|Business Administration|Fine Arts|Architecture|Applied Science))?|'
        r'Master\'s(?:\s+Degree)?|'
        r'Bachelor(?:\s+of\s+(?:Science|Arts|Engineering|Technology|Fine Arts|Business Administration|Applied Science))?|'
        r'Bachelor\'s(?:\s+Degree)?|'
        r'Doctor(?:\s+of\s+(?:Philosophy|Science|Medicine))?|'
        r'B\.?S\.?c?|B\.?Tech|B\.?E\.?|B\.?A\.?|'
        r'M\.?S\.?c?|M\.?Tech|M\.?B\.?A\.?|'
        r'Ph\.?D\.?|Associate(?:\s+of\s+(?:Science|Arts|Applied Science))?|Diploma)\b',
        re.IGNORECASE,
    )

    _GPA_PATTERN = re.compile(
        r'(?:(?:CGPA|GPA|Grade)[\s:]*([0-9]+(?:\.[0-9]+)?(?:\s*%)?(?:\s*\/\s*[0-9]+(?:\.[0-9]+)?)?))|'
        r'([0-9]\.[0-9]{1,2}\s*(?:\/\s*4(?:\.0)?)?\s*GPA)|'
        r'([0-9]{2}(?:\.[0-9]+)?\s*%)',
        re.IGNORECASE,
    )

    _DATE_RANGE_PATTERN = re.compile(
        r'\b(?P<start>(?:19|20)\d{2})\s*(?:[-–—to]+)\s*(?P<end>(?:19|20)\d{2}|Present|Current)\b',
        re.IGNORECASE,
    )

    _INSTITUTION_KEYWORDS = (
        "university", "college", "institute", "polytechnic", "academy", "school"
    )

    _INSTITUTION_ACRONYMS = {
        "MIT", "UCLA", "CMU", "NYU", "IIT", "NUS", "ETH", "CALTECH",
        "GEORGIA TECH", "UC BERKELEY", "UIUC", "USC", "UT AUSTIN", "BITS"
    }

    @classmethod
    def extract(cls, text: str) -> list[EducationRecord]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        raw_blocks: list[str] = []
        current_lines: list[str] = []

        for line in lines:
            cl = line.strip()
            if not cl:
                if current_lines:
                    raw_blocks.append("\n".join(current_lines))
                    current_lines = []
                continue

            # Split when a subsequent line introduces another degree
            if current_lines and cls._DEGREE_PATTERN.search(cl):
                if any(cls._DEGREE_PATTERN.search(prev) for prev in current_lines):
                    raw_blocks.append("\n".join(current_lines))
                    current_lines = []

            current_lines.append(cl)

        if current_lines:
            raw_blocks.append("\n".join(current_lines))

        records: list[EducationRecord] = []

        for block in raw_blocks:
            degree: Optional[str] = None
            field_of_study: Optional[str] = None
            institution: Optional[str] = None
            start_date: Optional[str] = None
            end_date: Optional[str] = None
            grade_or_cgpa: Optional[str] = None

            # 1. Degree match
            deg_match = cls._DEGREE_PATTERN.search(block)
            if deg_match:
                degree = deg_match.group(0).strip()

            # 2. Field of study heuristic (often after 'in' e.g. "Computer Science")
            field_match = re.search(r'\bin\s+([A-Za-z &]+?)(?=[,\n\(\)]|$)', block, re.IGNORECASE)
            if field_match:
                candidate_field = field_match.group(1).strip()
                if not any(w in candidate_field.lower() for w in ["university", "college", "institute"]):
                    field_of_study = candidate_field

            # 3. Institution match
            for line in block.splitlines():
                cl = line.strip()
                if not cl:
                    continue

                # If line is purely a degree declaration without institution mention, skip it
                if (
                    any(kw in cl.lower() for kw in ["bachelor", "master", "doctor", "ph.d", "diploma", "associate"])
                    and not any(kw in cl.lower() for kw in cls._INSTITUTION_KEYWORDS)
                    and not any(re.search(rf'\b{re.escape(ac)}\b', cl, re.IGNORECASE) for ac in cls._INSTITUTION_ACRONYMS)
                ):
                    continue

                # If line is purely a GPA or grade line, skip it
                if cls._GPA_PATTERN.search(cl) and not any(kw in cl.lower() for kw in cls._INSTITUTION_KEYWORDS):
                    continue

                inst_candidate = re.sub(r'[\(\[].*?[\)\]]', '', cl)
                inst_candidate = re.sub(r'\b(?:19|20)\d{2}\b.*$', '', inst_candidate)
                inst_candidate = inst_candidate.strip(" \t|,-—–•●")

                # If line had a degree prefix (e.g. "B.S. at Stanford University"), remove degree prefix
                if degree and degree.lower() in inst_candidate.lower():
                    inst_candidate = re.sub(
                        rf'(?<![A-Za-z0-9]){re.escape(degree)}\.?(?![A-Za-z0-9])(?:\s+(?:in|of)\s+[A-Za-z &]+)?(?:\s+(?:at|from|,|-|\|)\s+)?',
                        '',
                        inst_candidate,
                        flags=re.IGNORECASE,
                    ).strip(" \t|,-—–•●")

                # Try segment split if pipe is used (e.g. "MIT | 2017 - 2021" or "CMU | Pittsburgh, PA")
                parts = [p.strip(" \t|,-—–•●") for p in inst_candidate.split("|") if p.strip(" \t|,-—–•●")]
                matched_inst = None
                for p in parts:
                    if (
                        any(kw in p.lower() for kw in cls._INSTITUTION_KEYWORDS)
                        or p.upper() in cls._INSTITUTION_ACRONYMS
                        or any(re.search(rf'\b{re.escape(ac)}\b', p, re.IGNORECASE) for ac in cls._INSTITUTION_ACRONYMS)
                    ):
                        matched_inst = p
                        break

                if matched_inst and not any(kw in matched_inst.lower() for kw in ["bachelor", "master", "doctor", "ph.d"]):
                    institution = matched_inst
                    break

            # 4. Dates
            date_match = cls._DATE_RANGE_PATTERN.search(block)
            if date_match:
                start_date = date_match.group("start")
                end_date = date_match.group("end")
            else:
                years = re.findall(r'\b(?:19|20)\d{2}\b', block)
                if len(years) >= 2:
                    start_date = years[0]
                    end_date = years[1]
                elif len(years) == 1:
                    end_date = years[0]

            # 5. GPA / Grade
            gpa_match = cls._GPA_PATTERN.search(block)
            if gpa_match:
                grade_or_cgpa = gpa_match.group(0).strip()

            if degree or institution or field_of_study:
                is_ambiguous = (degree is None or institution is None)
                records.append(
                    EducationRecord(
                        degree=degree,
                        field_of_study=field_of_study,
                        institution=institution,
                        start_date=start_date,
                        end_date=end_date,
                        grade_or_cgpa=grade_or_cgpa,
                        original_text=block,
                        confidence="AMBIGUOUS" if is_ambiguous else "CONFIRMED",
                        is_ambiguous=is_ambiguous,
                    )
                )

        return records


# ---------------------------------------------------------------------------
# Project Extractor
# ---------------------------------------------------------------------------

class ProjectExtractor:
    """Extracts projects, descriptions, tech stacks, and URLs with ambiguity tracking."""

    @classmethod
    def extract(cls, text: str) -> list[ProjectRecord]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        project_blocks: list[tuple[str, list[str]]] = []

        current_title: Optional[str] = None
        current_desc: list[str] = []

        def commit_project():
            nonlocal current_title, current_desc
            if current_title:
                project_blocks.append((current_title, current_desc))

        for line in lines:
            cl = line.strip()
            if not cl:
                continue

            is_bullet = cl.startswith(("-", "*", "•", "●"))
            stripped_bullet = re.sub(r'^(?:[\-\*\•\●]\s*)', '', cl).strip()

            meta_prefixes = {
                "repository", "repo", "github", "gitlab", "url", "link",
                "demo", "live", "website", "technologies", "tech", "tools", "stack"
            }

            if (
                ":" in stripped_bullet
                and not stripped_bullet.lower().startswith(("http://", "https://", "ftp://"))
                and stripped_bullet.split(":", 1)[0].strip().lower() not in meta_prefixes
                and (is_bullet or len(stripped_bullet.split(":")[0].split()) <= 5)
            ):
                name_part, desc_part = stripped_bullet.split(":", 1)
                commit_project()
                current_title = name_part.strip()
                current_desc = [desc_part.strip()] if desc_part.strip() else []
            elif "|" in stripped_bullet and len(stripped_bullet.split("|")[0].split()) <= 6:
                name_part, desc_part = stripped_bullet.split("|", 1)
                commit_project()
                current_title = name_part.strip()
                current_desc = [desc_part.strip()] if desc_part.strip() else []
            else:
                if current_title is None:
                    current_title = stripped_bullet
                    current_desc = []
                else:
                    current_desc.append(stripped_bullet)

        commit_project()

        records: list[ProjectRecord] = []
        for name, desc_lines in project_blocks:
            desc_text = " ".join(desc_lines).strip()
            combined_text = f"{name} {desc_text}"

            # 1. Extract Project URL directly
            url_match = URL_EXCLUSION_PATTERN.search(combined_text)
            proj_url = url_match.group(0).rstrip(".,)") if url_match else None

            # 2. Exclude URL from tech stack scan to prevent false positives (e.g. github.com -> Git)
            text_without_urls = URL_EXCLUSION_PATTERN.sub(" ", combined_text)

            # 3. Extract technologies explicitly mentioned in project text
            techs: list[str] = []
            for key, (canonical, _) in _SKILL_LOOKUP.items():
                if canonical not in techs:
                    pattern = rf'(?<![a-zA-Z0-9]){re.escape(key)}(?![a-zA-Z0-9])'
                    if re.search(pattern, text_without_urls, re.IGNORECASE):
                        techs.append(canonical)

            full_block = f"{name}: {desc_text}".strip() if desc_text else name
            is_ambiguous = (len(techs) == 0)

            records.append(
                ProjectRecord(
                    project_name=name,
                    description=desc_text,
                    technologies=techs,
                    project_url=proj_url,
                    original_text=full_block,
                    confidence="AMBIGUOUS" if is_ambiguous else "CONFIRMED",
                    is_ambiguous=is_ambiguous,
                )
            )

        return records


# ---------------------------------------------------------------------------
# Top-Level Entity Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Main entity extraction orchestrator.
    Consumes ParsedResumeSections and produces validated ResumeEntities.
    """

    def __init__(self) -> None:
        self.skills_extractor = SkillsExtractor()
        self.experience_extractor = ExperienceExtractor()
        self.education_extractor = EducationExtractor()
        self.project_extractor = ProjectExtractor()

    def extract_entities(self, parsed_resume: ParsedResumeSections) -> ResumeEntities:
        """
        Extract all entities across sections deterministically.

        Args:
            parsed_resume: The structured section model from Phase 2.2.

        Returns:
            ResumeEntities: Validated entity model containing skills, experience, education, projects.
        """
        all_skills: list[ExtractedSkill] = []
        all_experience: list[WorkExperience] = []
        all_education: list[EducationRecord] = []
        all_projects: list[ProjectRecord] = []

        seen_skills: set[str] = set()

        for sec in parsed_resume.sections:
            if sec.section_type == SectionType.TECHNICAL_SKILLS:
                skills = self.skills_extractor.extract_from_skills_section(sec.content)
                for s in skills:
                    if s.name not in seen_skills:
                        seen_skills.add(s.name)
                        all_skills.append(s)

            elif sec.section_type in (SectionType.WORK_EXPERIENCE, SectionType.INTERNSHIPS):
                jobs = self.experience_extractor.extract(sec.content)
                all_experience.extend(jobs)
                # Scan for unlisted skills in experience
                for j in jobs:
                    mentioned = self.skills_extractor.scan_mentioned_skills(
                        j.original_text,
                        source_section="Work Experience",
                        already_extracted=seen_skills,
                    )
                    for m in mentioned:
                        if m.name not in seen_skills:
                            seen_skills.add(m.name)
                            all_skills.append(m)

            elif sec.section_type in (SectionType.PROFESSIONAL_SUMMARY, SectionType.CAREER_OBJECTIVE):
                # Scan for technical skills mentioned in candidate professional summary/objective
                mentioned = self.skills_extractor.scan_mentioned_skills(
                    sec.content,
                    source_section="Summary",
                    already_extracted=seen_skills,
                )
                for m in mentioned:
                    if m.name not in seen_skills:
                        seen_skills.add(m.name)
                        all_skills.append(m)

            elif sec.section_type == SectionType.EDUCATION:
                edu_records = self.education_extractor.extract(sec.content)
                all_education.extend(edu_records)

            elif sec.section_type == SectionType.PROJECTS:
                projs = self.project_extractor.extract(sec.content)
                all_projects.extend(projs)
                # Scan for unlisted skills in projects
                for p in projs:
                    mentioned = self.skills_extractor.scan_mentioned_skills(
                        p.original_text,
                        source_section="Projects",
                        already_extracted=seen_skills,
                    )
                    for m in mentioned:
                        if m.name not in seen_skills:
                            seen_skills.add(m.name)
                            all_skills.append(m)

        return ResumeEntities(
            skills=all_skills,
            experience=all_experience,
            education=all_education,
            projects=all_projects,
        )


def extract_resume_entities(parsed_resume: ParsedResumeSections) -> ResumeEntities:
    """Convenience functional wrapper for EntityExtractor.extract_entities."""
    extractor = EntityExtractor()
    return extractor.extract_entities(parsed_resume)
