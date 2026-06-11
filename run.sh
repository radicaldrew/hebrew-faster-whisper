#!/usr/bin/env bash
# HF_TOKEN is required at build time for pyannote model download.
# Export it in your shell first — never hardcode it here.
docker build --build-arg HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment}" -t whisper-worker:latest .
