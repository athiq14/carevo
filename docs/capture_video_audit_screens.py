"""
Capture verification screenshots for Video Review Bug Fixes & Product Integrity Audit.
Specifically captures:
1. Resume Projects Tab (rendered safely without AttributeError).
2. Job Matching Results (score-aware, provisional explanation, coverage vs completeness).
3. Career Advisor (personalized skill gaps connected to target role or labeled preview).
4. LinkedIn Analysis (privacy notice, input fields, and analysis).
5. Settings (authoritative 45/25/15/15 Phase 4 weights from WeightManager).
6. Agentic Workflow Empty State (explanation of snapshot requirement and button to JD Matching).
"""
import time
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

LOCAL_DIR = Path("docs/screenshots")
ARTIFACT_DIR = Path(r"C:\Users\mathi\OneDrive\Documents\ai-career-intelligence-agent\docs\screenshots")
BRAIN_ARTIFACT_DIR = Path(r"C:\Users\mathi\.gemini\antigravity\brain\eef32a5e-39cb-49d0-9f80-c31491bcd931\screenshots")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def snap(page, filename, delay=1.5):
    time.sleep(delay)
    loc_path = LOCAL_DIR / f"{filename}.png"
    art_path = BRAIN_ARTIFACT_DIR / f"{filename}.png"
    page.screenshot(path=str(loc_path), full_page=False)
    shutil.copy(loc_path, art_path)
    print(f"[OK] Saved {filename}.png ({loc_path.stat().st_size} bytes)")

def select_nav(page, name):
    print(f"Navigating to: {name}")
    labels = page.locator('[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stRadioOption"]')
    labels.filter(has_text=name).first.click()
    time.sleep(2)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 920})
        page = context.new_page()

        print("Navigating to application at http://localhost:8501...")
        page.goto("http://localhost:8501")
        page.wait_for_selector(".stApp", timeout=15000)
        time.sleep(2)

        # -------------------------------------------------------------
        # 1. Resume Projects Tab (Issue 1)
        # -------------------------------------------------------------
        select_nav(page, "Resume Analysis")
        # Ensure sample candidate is loaded
        sample_btn = page.locator('button:has-text("Load Sample Candidate")')
        if sample_btn.count() > 0:
            sample_btn.first.click()
            time.sleep(2)
        analyze_btn = page.locator('button:has-text("Analyze Resume →")')
        if analyze_btn.count() > 0:
            analyze_btn.first.click()
            time.sleep(3)
        # Click "Projects" tab
        projects_tab = page.locator('button[data-baseweb="tab"]:has-text("Projects")')
        if projects_tab.count() > 0:
            projects_tab.first.click()
            time.sleep(1.5)
        snap(page, "corrected_01_resume_projects_tab")

        # -------------------------------------------------------------
        # 2. Job Matching Results (Issue 2)
        # -------------------------------------------------------------
        select_nav(page, "JD Matching")
        # Pre-fill with Web Analyst or Stripe
        web_btn = page.locator('button:has-text("Web Analyst")')
        if web_btn.count() > 0:
            web_btn.first.click()
            time.sleep(1)
        # Run match
        match_btn = page.locator('button:has-text("Analyze Match →")')
        if match_btn.count() > 0:
            match_btn.first.click()
            time.sleep(3)
        page.evaluate("window.scrollTo(0, 400)")
        snap(page, "corrected_02_job_matching_results")

        # -------------------------------------------------------------
        # 3. Career Advisor (Issue 3)
        # -------------------------------------------------------------
        select_nav(page, "Career Advisor")
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "corrected_03_career_advisor")

        # -------------------------------------------------------------
        # 4. LinkedIn Analysis (Issue 4)
        # -------------------------------------------------------------
        select_nav(page, "LinkedIn Analysis")
        # Click Load Sample Profile Data
        li_sample_btn = page.locator('button:has-text("Load Sample Profile Data")')
        if li_sample_btn.count() > 0:
            li_sample_btn.first.click()
            time.sleep(1.5)
        # Click Analyze Profile Content
        li_analyze_btn = page.locator('button:has-text("Analyze Profile Content")')
        if li_analyze_btn.count() > 0:
            li_analyze_btn.first.click()
            time.sleep(2)
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "corrected_04_linkedin_analysis")

        # -------------------------------------------------------------
        # 5. Settings (Issue 5)
        # -------------------------------------------------------------
        select_nav(page, "Settings")
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "corrected_05_settings")

        # -------------------------------------------------------------
        # 6. Agentic Workflow Empty State (Issue 6)
        # -------------------------------------------------------------
        # First reset session to test clean empty state
        select_nav(page, "Home")
        # Clear matching state via Developer Diagnostics or Reset
        expander = page.locator('[data-testid="stSidebar"] [data-testid="stExpander"]')
        expander.click()
        time.sleep(1)
        reset_all_btn = page.locator('button:has-text("Clear Session State & Cache")')
        if reset_all_btn.count() > 0:
            reset_all_btn.first.click()
            time.sleep(2)
        select_nav(page, "Agents (Workflow)")
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "corrected_06_agentic_workflow_empty_state")

        browser.close()
        print("All audit verification screenshots captured successfully!")

if __name__ == "__main__":
    run()

