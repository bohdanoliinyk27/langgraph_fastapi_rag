from fastapi import APIRouter, HTTPException
from models import ChatAskRequestV2, ChatAnswer, Message
from repo import repo
from services.chat_engine import ask_question
from services.template_service import TemplateService

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/{chat_id}/ask", response_model=ChatAnswer)
async def ask(chat_id: str, req: ChatAskRequestV2):
    chat = repo.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    history_pairs = [(m.role, str(m.content)) for m in chat.history]

    template_text = ""
    if req.template_id:
        tpl = TemplateService.get(req.template_id)
        if not tpl:
            raise HTTPException(status_code=404, detail="Template not found")
        template_text = tpl.get("content", "").strip()

    repo.append(chat_id, Message(role="user", content=req.question))

    final_question = f"{template_text}\n\n{req.question}" if template_text else req.question

    result = ask_question(
        final_question,
        req.attached or [],
        history_pairs,
        force_web=bool(req.use_web),
        force_pubmed=bool(req.use_pubmed),
    )

    answer = result.get("answer", "")
    trace = result.get("trace")

    repo.append(chat_id, Message(role="assistant", content=answer))
    return ChatAnswer(answer=answer, trace=trace)
