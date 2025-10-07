from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import files, chats, chat
from routers.templates import router as templates_router

app = FastAPI(title="Adaptive RAG Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(files.router)
app.include_router(templates_router)
app.include_router(chats.router)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)