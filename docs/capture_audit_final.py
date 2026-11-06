"""
Audit Screenshot Suite for AI Career Intelligence Agent.
Captures all 15 required visual and functional verification screens in high resolution.
"""
import time
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

LOCAL_DIR = Path("docs/screenshots")
ARTIFACT_DIR = Path(r"C:\Users\mathi\.gemini\antigravity\brain\eef32a5e-39cb-49d0-9f80-c31491bcd931\screenshots")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def snap(page, filename, delay=1.5):
    time.sleep(delay)
    loc_path = LOCAL_DIR / f"{filename}.png"
    art_path = ARTIFACT_DIR / f"{filename}.png"
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
        # 1. Home Dashboard
        # -------------------------------------------------------------
        select_nav(page, "Home")
        snap(page, "01_home_dashboard")

        # -------------------------------------------------------------
        # 2. Resume Upload Screen
        # -------------------------------------------------------------
        select_nav(page, "Resume Analysis")
        back_btn = page.locator('button:has-text("← Back")')
        if back_btn.count() > 0:
            back_btn.first.click()
            time.sleep(1.5)
        snap(page, "02_resume_upload_screen")

        # -------------------------------------------------------------
        # 3. Resume Analysis Results
        # -------------------------------------------------------------
        sample_btn = page.locator('button:has-text("Load Sample Candidate")')
        if sample_btn.count() > 0:
            sample_btn.first.click()
            time.sleep(3)
        analyze_btn = page.locator('button:has-text("Analyze Resume →")')
        if analyze_btn.count() > 0:
            analyze_btn.first.click()
            time.sleep(3)
        snap(page, "03_resume_analysis_results")

        # -------------------------------------------------------------
        # 4. Job Description Input Screen
        # -------------------------------------------------------------
        select_nav(page, "JD Matching")
        stripe_btn = page.locator('button:has-text("Stripe Platform")')
        if stripe_btn.count() > 0:
            stripe_btn.first.click()
            time.sleep(1)
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "04_jd_input_screen")

        # -------------------------------------------------------------
        # 5. Job Matching Results (Scrolled to reveal circular score & breakdown)
        # -------------------------------------------------------------
        match_btn = page.locator('button:has-text("Analyze Match →")')
        if match_btn.count() > 0:
            match_btn.first.click()
            print("Evaluating matching pipeline in browser...")
            time.sleep(6)
        # Scroll down to display the matching results hero card and dimensional breakdown
        page.evaluate("window.scrollTo(0, 520)")
        snap(page, "05_jd_matching_results")

        # -------------------------------------------------------------
        # 6. Career Advisor
        # -------------------------------------------------------------
        select_nav(page, "Career Advisor")
        snap(page, "06_career_advisor")

        # -------------------------------------------------------------
        # 7. Agentic Workflow Before Launch
        # -------------------------------------------------------------
        select_nav(page, "Agents (Workflow)")
        page.evaluate("window.scrollTo(0, 0)")
        # Click Reset if already in a session
        reset_btn = page.locator('button:has-text("Reset Workflow")')
        if reset_btn.count() > 0:
            reset_btn.first.click()
            time.sleep(2)
        snap(page, "07_workflow_before_launch")

        # -------------------------------------------------------------
        # 8. Workflow Clarification Checkpoint
        # -------------------------------------------------------------
        # In JD Matching, provide sparse criteria that triggers clarification
        select_nav(page, "JD Matching")
        jd_box = page.locator('textarea')
        jd_box.fill("Job Title: Cloud Architect\nRequired Skills: Kubernetes, Docker, Python, Redis.")
        time.sleep(1)
        page.locator('button:has-text("Analyze Match →")').first.click()
        print("Evaluating sparse requirements for clarification trigger...")
        time.sleep(6)

        # Back to Agents (Workflow)
        select_nav(page, "Agents (Workflow)")
        reset_btn = page.locator('button:has-text("Reset Workflow")')
        if reset_btn.count() > 0:
            reset_btn.first.click()
            time.sleep(2)

        page.locator('button:has-text("Launch Agentic Workflow")').first.click()
        print("Starting workflow coordinator...")
        time.sleep(5)
        page.evaluate("window.scrollTo(0, 250)")
        snap(page, "08_workflow_clarification")

        # -------------------------------------------------------------
        # 9. Goal Selection Checkpoint
        # -------------------------------------------------------------
        # Submit candidate clarification
        clar_area = page.locator('textarea')
        if clar_area.count() > 0:
            clar_area.fill("I have extensive practical production experience with Python microservices and Kubernetes container orchestration.")
            time.sleep(0.5)
            sub_clar = page.locator('button:has-text("Submit Clarification & Resume Workflow")')
            if sub_clar.count() > 0:
                sub_clar.first.click()
                time.sleep(4)
        page.evaluate("window.scrollTo(0, 250)")
        snap(page, "09_workflow_goal_selection")

        # -------------------------------------------------------------
        # 10. Specialist Deliverables (Final Approval Checkpoint)
        # -------------------------------------------------------------
        confirm_goal = page.locator('button:has-text("Confirm Goal & Execute Specialists")')
        if confirm_goal.count() > 0:
            confirm_goal.first.click()
            print("Executing specialists and deterministic safety audit...")
            time.sleep(10)
        page.evaluate("window.scrollTo(0, 300)")
        snap(page, "10_workflow_specialist_deliverables")

        # -------------------------------------------------------------
        # 11. Safety Review Tab
        # -------------------------------------------------------------
        audit_tab = page.locator('button[role="tab"]:has-text("Safety & Score Audit")')
        if audit_tab.count() > 0:
            audit_tab.first.click()
            time.sleep(1.5)
        page.evaluate("window.scrollTo(0, 350)")
        snap(page, "11_workflow_safety_review")

        # -------------------------------------------------------------
        # 12. Final Approval Controls
        # -------------------------------------------------------------
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        snap(page, "12_workflow_final_approval")

        # -------------------------------------------------------------
        # 13. Completed Report
        # -------------------------------------------------------------
        approve_btn = page.locator('button:has-text("Approve & Synthesize Final Report")')
        if approve_btn.count() > 0:
            approve_btn.first.click()
            print("Synthesizing final intelligence report...")
            time.sleep(6)
        page.evaluate("window.scrollTo(0, 200)")
        snap(page, "13_workflow_completed_report")

        # -------------------------------------------------------------
        # 14. LinkedIn Analysis
        # -------------------------------------------------------------
        select_nav(page, "LinkedIn Analysis")
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "14_linkedin_analysis")

        # -------------------------------------------------------------
        # 15. History and Settings
        # -------------------------------------------------------------
        select_nav(page, "Settings")
        page.evaluate("window.scrollTo(0, 0)")
        snap(page, "15_history_and_settings")

        browser.close()
        print("All 15 verification screenshots successfully captured!")

if __name__ == "__main__":
    run()

