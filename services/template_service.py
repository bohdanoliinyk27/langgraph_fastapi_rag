# services/template_service.py
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

class TemplateService:
    @staticmethod
    def _path(tid: str) -> Path:
        return TEMPLATES_DIR / f"{tid}.json"

    @staticmethod
    def create(name: str, content: str) -> Dict[str, Any]:
        tid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        data = {
            "id": tid,
            "name": name,
            "content": content,
            "created_at": now,
            "updated_at": now,
        }
        with TemplateService._path(tid).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    @staticmethod
    def update(tid: str, name: Optional[str], content: Optional[str]) -> Optional[Dict[str, Any]]:
        p = TemplateService._path(tid)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        if name is not None:
            data["name"] = name
            changed = True
        if content is not None:
            data["content"] = content
            changed = True

        if changed:
            data["updated_at"] = datetime.utcnow().isoformat()
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    @staticmethod
    def get(tid: str) -> Optional[Dict[str, Any]]:
        p = TemplateService._path(tid)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def delete(tid: str) -> bool:
        p = TemplateService._path(tid)
        if not p.exists():
            return False
        p.unlink()
        return True

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        res: List[Dict[str, Any]] = []
        for file in TEMPLATES_DIR.glob("*.json"):
            try:
                with file.open("r", encoding="utf-8") as f:
                    res.append(json.load(f))
            except Exception:
                continue
        # latest first
        res.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return res
