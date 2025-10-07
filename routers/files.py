from fastapi import APIRouter, UploadFile, File, HTTPException
from models import UploadResult, FileInfo
from deps import UPLOAD_DIR, SUMMARY_DIR
from services.create_index import ingest_file

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=UploadResult)
async def upload_file(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf or .txt supported")

    dest = UPLOAD_DIR / file.filename
    data = await file.read()
    dest.write_bytes(data)

    # Dummy "ingestion"
    report = ingest_file(str(dest), file.filename)

    # Naive summary: take first N bytes of text if .txt, else save placeholder
    summary_text = ""
    try:
        if file.filename.endswith(".txt"):
            raw = dest.read_text(encoding="utf-8", errors="ignore")
            summary_text = raw[:2000]
        else:
            summary_text = f"Summary placeholder for PDF '{file.filename}'."
    except Exception:
        summary_text = "Summary unavailable."

    SUMMARY_DIR.mkdir(exist_ok=True)
    (SUMMARY_DIR / f"{dest.stem}.txt").write_text(summary_text, encoding="utf-8")

    return UploadResult(message=report, namespace=file.filename)

@router.get("/list", response_model=list[FileInfo])
async def list_files():
    out = []
    for p in UPLOAD_DIR.glob("*"):
        if p.is_file():
            out.append(FileInfo(name=p.name, size=p.stat().st_size))
    return out

@router.delete("/{filename}")
async def delete_file(filename: str):
    f = UPLOAD_DIR / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail="File not found")
    f.unlink()
    s = SUMMARY_DIR / f"{f.stem}.txt"
    if s.exists():
        s.unlink()
    return {"ok": True}
