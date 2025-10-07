from fastapi import APIRouter
from models import ChatCreate, ChatSummary
from repo import repo

router = APIRouter(prefix="/chats", tags=["chats"])

@router.post("", response_model=ChatSummary)
async def create_chat(payload: ChatCreate):
    chat = repo.create(payload)
    return ChatSummary(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        messages_count=len(chat.history),
    )

@router.get("", response_model=list[ChatSummary])
async def list_chats():
    items = []
    for c in repo.list():
        items.append(ChatSummary(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
            messages_count=len(c.history),
        ))
    return items
