import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException, status

# ======================================================
# Config
# ======================================================
# All technician-uploaded diagnostic images live under this directory.
# Nothing outside of it should ever be reachable via a client-supplied value.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def save_upload_file(file: UploadFile) -> str:
    """
    Persists an uploaded file under UPLOAD_DIR using a freshly generated UUID
    as the filename (extension preserved only if it's an allow-listed image
    type). Returns the file_id (UUID string) the client should pass back as
    `attached_image_path` on subsequent chat requests.

    The original client-supplied filename is never used to construct a path,
    so there is no path-traversal surface here regardless of what the client
    sends as `file.filename`.
    """
    original_ext = Path(file.filename or "").suffix.lower()
    if original_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{original_ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    file_id = str(uuid.uuid4())
    dest_path = UPLOAD_DIR / f"{file_id}{original_ext}"

    bytes_written = 0
    try:
        with dest_path.open("wb") as out_file:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    out_file.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds max upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
                    )
                out_file.write(chunk)
    except HTTPException:
        raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded file."
        )
    finally:
        file.file.close()

    return file_id


def resolve_upload_path(file_id: str) -> Optional[Path]:
    """
    Resolves a client-supplied file_id (expected to be a bare UUID, as
    returned by save_upload_file) to an actual on-disk path, WITHOUT ever
    trusting client input as a literal path.

    Returns None if file_id is not a well-formed UUID, if no matching file
    exists, or if the resolved path somehow escapes UPLOAD_DIR (defense in
    depth against symlink / traversal tricks).
    """
    if not file_id or not _UUID_RE.match(file_id):
        return None

    matches = sorted(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        return None

    resolved = matches[0].resolve()
    if resolved.parent != UPLOAD_DIR:
        return None

    return resolved