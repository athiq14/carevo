"""
Unit tests to verify initial project dependencies import and version integrity.
"""


def test_import_streamlit():
    import streamlit as st
    assert hasattr(st, "set_page_config")


def test_import_pandas():
    import pandas as pd
    df = pd.DataFrame({"col": [1, 2, 3]})
    assert len(df) == 3


def test_import_pymupdf():
    import pymupdf
    assert hasattr(pymupdf, "open")


def test_import_dotenv():
    import dotenv
    assert hasattr(dotenv, "load_dotenv")


def test_import_pydantic():
    from pydantic import BaseModel
    class SampleModel(BaseModel):
        name: str
    m = SampleModel(name="test")
    assert m.name == "test"


def test_import_google_genai():
    from google import genai
    from google.genai import types
    assert hasattr(genai, "Client")
    assert hasattr(types, "GenerateContentConfig")


def test_import_groq():
    import groq
    from groq import Groq
    assert hasattr(groq, "Groq")
