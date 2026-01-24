"""Placeholder REST API module."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def status():
    return {"status": "ok"}
