import os
from dotenv import load_dotenv
from models import Chat, ChatCreate, Message
from pymongo import MongoClient
import uuid
from datetime import datetime
from typing import List, Optional

load_dotenv()

class ChatRepository:
    def __init__(self, mongo_uri: str, db_name: str = "chatdb"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["chats"]

    def create(self, data: ChatCreate) -> Chat:
        chat_id = str(uuid.uuid4())
        now = datetime.utcnow()
        chat = Chat(
            id=chat_id,
            title=data.title or f"Chat {chat_id[:8]}",
            created_at=now,
            updated_at=now,
            history=[]
        )
        self.collection.insert_one(chat.dict())
        return chat

    def list(self) -> List[Chat]:
        docs = self.collection.find().sort("updated_at", -1)
        return [Chat(**doc) for doc in docs]

    def get(self, chat_id: str) -> Optional[Chat]:
        doc = self.collection.find_one({"id": chat_id})
        return Chat(**doc) if doc else None

    def append(self, chat_id: str, message: Message) -> Optional[Chat]:
        now = datetime.utcnow()
        self.collection.update_one(
            {"id": chat_id},
            {
                "$push": {"history": message.dict()},
                "$set": {"updated_at": now},
            }
        )
        return self.get(chat_id)

MONGO_URI = os.getenv("MONGO_URI")
repo = ChatRepository(MONGO_URI)
