"""OpenAI-compatible front door for the FakeDet system.

open-webui (or any OpenAI client) talks to THIS service; it runs the
Haystack pipeline and the pipeline reaches the trained model backend.
"""
