from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from services.template_service import TemplateService

router = APIRouter(prefix="/templates", tags=["templates"])

class TemplateCreate(BaseModel):
    name: str = Field(..., max_length=200)
    content: str = Field(..., max_length=20000)

class TemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    content: Optional[str] = Field(None, max_length=20000)

class TemplateOut(BaseModel):
    id: str
    name: str
    content: str
    created_at: str
    updated_at: str

@router.post("", response_model=TemplateOut)
def create_template(body: TemplateCreate):
    return TemplateService.create(body.name, body.content)

@router.get("", response_model=List[TemplateOut])
def list_templates():
    return TemplateService.list_all()

@router.get("/{template_id}", response_model=TemplateOut)
def get_template(template_id: str):
    data = TemplateService.get(template_id)
    if not data:
        raise HTTPException(status_code=404, detail="Template not found")
    return data

@router.patch("/{template_id}", response_model=TemplateOut)
def update_template(template_id: str, body: TemplateUpdate):
    data = TemplateService.update(template_id, body.name, body.content)
    if not data:
        raise HTTPException(status_code=404, detail="Template not found")
    return data

@router.delete("/{template_id}")
def delete_template(template_id: str):
    ok = TemplateService.delete(template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"ok": True}
