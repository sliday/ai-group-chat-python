# main.py
import os
import requests
import json
import logging
from uuid import uuid4
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime

# --- Configuration & Setup ---
load_dotenv()  # Load environment variables from .env file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Group AI Chat")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:8000") # Default if not set
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "Simple AI Group Chat") # Default if not set
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    logger.error("CRITICAL: OPENROUTER_API_KEY environment variable not set. API calls will fail.")
    # Optionally exit or disable API features
    # exit(1)

# --- In-Memory Storage ---
# Stores chat history for each room_id
# Format: chats[room_id] = [{"role": "user" | "assistant", "content": "...", "username": "...", "timestamp": "ISO_FORMAT"}, ...]
chats: Dict[str, List[Dict[str, Any]]] = {}

# --- Models ---
class MessagePayload(BaseModel):
    message: str
    username: Optional[str] = "User" # Default username if not provided

# --- Helper Functions ---
def get_openrouter_headers() -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if YOUR_SITE_URL:
        headers["HTTP-Referer"] = YOUR_SITE_URL
    if YOUR_SITE_NAME:
        headers["X-Title"] = YOUR_SITE_NAME
    return headers

# --- API Endpoints ---

# Serve Frontend
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page."""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        logger.error("index.html not found")
        raise HTTPException(status_code=500, detail="Frontend file missing.")

@app.get("/chat/{room_id}", response_class=HTMLResponse)
async def get_chat_room_page(room_id: str, request: Request):
    """Serves the HTML page for a specific chat room, ensuring room exists."""
    if room_id not in chats:
         # Redirect to home page with an error query param? Or just show 404?
         # Showing 404 is simpler API-wise. Frontend can handle redirect if needed.
        logger.warning(f"Attempted to access non-existent room: {room_id}")
        # Return a simple message or redirect - returning index allows JS check
        # return HTMLResponse("Invalid room ID. <a href='/'>Create a new room?</a>", status_code=404)
        # Serving index.html allows JS to potentially handle joining via URL more gracefully
        try:
            return FileResponse("index.html")
        except FileNotFoundError:
            logger.error("index.html not found")
            raise HTTPException(status_code=500, detail="Frontend file missing.")
    # Room exists, serve the main page - JS will handle the rest
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        logger.error("index.html not found")
        raise HTTPException(status_code=500, detail="Frontend file missing.")

# Room Management
@app.post("/api/create_room")
async def create_room():
    """Creates a new chat room and returns its unique ID."""
    room_id = str(uuid4())[:8]  # Shorter, easier to share UUID
    while room_id in chats: # Ensure uniqueness, although collision is unlikely
        room_id = str(uuid4())[:8]
        
    chats[room_id] = []  # Initialize empty chat history
    logger.info(f"Created room: {room_id}")
    return {"room_id": room_id}

# Chat History (for polling and initial load)
@app.get("/api/chat/{room_id}/messages")
async def get_messages(room_id: str):
    """Returns the current full chat history for a given room."""
    if room_id not in chats:
        logger.warning(f"Attempted fetch messages for non-existent room: {room_id}")
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Return the stored history including metadata (username, timestamp)
    return {"messages": chats.get(room_id, [])}

# Send Message & Get AI Response
@app.post("/api/chat/{room_id}/message")
async def post_message(room_id: str, payload: MessagePayload):
    """Receives a user message, adds it to history, gets AI response, adds it too, returns AI response."""
    if room_id not in chats:
        logger.warning(f"Attempted post message to non-existent room: {room_id}")
        raise HTTPException(status_code=404, detail="Room not found")
        
    if not OPENROUTER_API_KEY:
         raise HTTPException(status_code=503, detail="AI Service not configured by administrator.")

    username = payload.username if payload.username else "User"
    user_message_content = payload.message.strip()

    if not user_message_content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    timestamp = datetime.now().isoformat()

    # Add user message to history (with metadata)
    user_message_entry = {
        "role": "user",
        "content": user_message_content,
        "username": username,
        "timestamp": timestamp
    }
    chats[room_id].append(user_message_entry)
    logger.info(f"Room {room_id} - User '{username}': {user_message_content}")

    # Prepare messages for OpenRouter API (strip our metadata)
    api_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in chats[room_id] # Send the full history for context
        if msg.get("role") in ["user", "assistant"] # Ensure only valid roles sent
    ]

    # --- Call OpenRouter API ---
    try:
        api_payload = {
            "model": "openrouter/auto", # Use auto model selection
            "messages": api_messages
        }
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=get_openrouter_headers(),
            json=api_payload,
            timeout=35 # Slightly longer timeout for AI generation
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        api_response = response.json()

        # --- Process Response ---
        if "choices" in api_response and len(api_response["choices"]) > 0:
            assistant_message_data = api_response["choices"][0].get("message", {})
            assistant_reply_content = assistant_message_data.get("content", "").strip()

            if not assistant_reply_content:
                 assistant_reply_content = "Sorry, I received an empty response."
                 logger.warning(f"Room {room_id} - Received empty AI response. API Raw: {api_response}")
            
            # Add assistant response to history (with metadata)
            assistant_timestamp = datetime.now().isoformat()
            assistant_message_entry = {
                "role": assistant_message_data.get("role", "assistant"),
                "content": assistant_reply_content,
                "username": "AI Assistant",
                "timestamp": assistant_timestamp
            }
            chats[room_id].append(assistant_message_entry)
            logger.info(f"Room {room_id} - AI: {assistant_reply_content[:80]}...") # Log snippet

            # Return only the AI's message details to the frontend that called POST
            return JSONResponse(content={
                "reply": assistant_reply_content,
                "username": "AI Assistant",
                "timestamp": assistant_timestamp
            })
        else:
            logger.error(f"Room {room_id} - Unexpected API response structure: {api_response}")
            # Don't add a failed AI response to history. User message remains.
            raise HTTPException(status_code=500, detail="Invalid response structure from AI service.")

    except requests.exceptions.Timeout:
        logger.error(f"Room {room_id} - API request timed out.")
        # Don't add a failed AI response to history. User message remains.
        raise HTTPException(status_code=504, detail="AI service request timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Room {room_id} - API request failed: {e}")
        # Don't add a failed AI response to history. User message remains.
        raise HTTPException(status_code=503, detail=f"AI service unavailable: {e}")
    except Exception as e:
        logger.exception(f"Room {room_id} - Error processing message: {e}") # Log full traceback
        # Don't add a failed AI response to history. User message remains.
        raise HTTPException(status_code=500, detail="Internal server error processing message.")

# --- Run Server (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8000')}")
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True # Reload enabled for development
    )