"""DOCX -> PDF through headless LibreOffice, so Word files share the PDF path."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from app.config import get_settings
from app.pipeline.errors import ExtractionError


def libreoffice_available() -> bool:
    return shutil.which("soffice") is not None


def docx_to_pdf(data: bytes) -> bytes:
    if not libreoffice_available():
        raise RuntimeError("LibreOffice (soffice) is not installed; DOCX support unavailable")
    with tempfile.TemporaryDirectory(prefix="docx2pdf-") as tmp:
        src = Path(tmp, "in.docx")
        src.write_bytes(data)
        cmd = [
            "soffice",
            "--headless",
            "--norestore",
            "--nolockcheck",
            "--nodefault",
            f"-env:UserInstallation=file://{tmp}/profile",  # private profile => safe to run concurrently
            "--convert-to",
            "pdf",
            "--outdir",
            tmp,
            str(src),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=get_settings().libreoffice_timeout_s, check=True)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError("DOCX conversion timed out") from e
        except subprocess.CalledProcessError as e:
            raise ExtractionError("Could not convert the DOCX file") from e
        out = Path(tmp, "in.pdf")
        if not out.exists():
            raise ExtractionError("Could not convert the DOCX file")
        return out.read_bytes()
