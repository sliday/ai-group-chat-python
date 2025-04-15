# main.py
import os
import httpx
import json
import logging
from uuid import uuid4
from typing import Dict, List, Any, Optional, Set
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import asyncio

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

# Store active users per room
# Format: active_users[room_id] = {username1, username2, ...}
active_users: Dict[str, Set[str]] = {}

# Store banned users per room (now using fingerprints)
banned_fingerprints: Dict[str, Set[str]] = {}

# Store username to fingerprint mapping for moderation
user_fingerprints: Dict[str, Dict[str, str]] = {}  # room_id -> {username: fingerprint}

# --- Models ---
class MessagePayload(BaseModel):
    message: str
    username: Optional[str] = "User" # Default username if not provided
    fingerprint: str

class UserPresencePayload(BaseModel):
    username: str
    action: str  # "join" or "leave"
    fingerprint: str

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

def check_message_moderation(message: str, username: str) -> Dict[str, Any]:
    """Check if a message is appropriate using AI and explicit word filtering."""
    # First, check for explicit profanity using a simple word list
    explicit_profanity = {
        "fuck", "shit", "bitch", "cunt", "dick", "pussy", "asshole", "bastard",
        # Add variations
        "fuckyou", "fuk", "fucker", "fucking", "fuckin", "fck", "stfu"
    }
    
    # Convert to lowercase and remove spaces for checking
    message_normalized = message.lower().replace(" ", "")
    
    # Check for explicit profanity first
    for word in explicit_profanity:
        if word in message_normalized:
            return {
                "is_inappropriate": True,
                "reason": "explicit profanity",
                "action": "ban"  # Immediate ban for explicit profanity
            }

    # If no explicit profanity found, proceed with AI moderation
    moderation_prompt = [
        {"role": "system", "content": """You are a strict chat moderator. Message is inappropriate if it contains ANY of:

1. Indirect profanity or offensive language
2. Personal attacks or insults
3. Hate speech (racism, sexism, etc.)
4. Threats or intimidation
5. Sexual content or harassment
6. Spam or flooding
7. Personal information sharing
8. Scams or phishing attempts
9. Extremist content
10. Drug trafficking
11. Violence promotion
12. Deliberate misinformation

If inappropriate, respond with reason in max 10 tokens.
If message is fine, respond with "0".

Examples of inappropriate content:
- Coded profanity or leetspeak
- Suggestive or inappropriate innuendos
- Passive-aggressive insults
- Hostile or aggressive behavior
"""},
        {"role": "user", "content": message}
    ]

    try:
        response = httpx.post(
            url=OPENROUTER_API_URL,
            headers=get_openrouter_headers(),
            json={
                "model": "openai/gpt-4o-mini",
                "messages": moderation_prompt,
                "max_tokens": 10,
                "temperature": 0.1  # Lower temperature for more consistent moderation
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            moderation_result = result["choices"][0]["message"]["content"].strip()
            
            # Simple response parsing
            if moderation_result == "0":
                return {"is_inappropriate": False, "reason": None, "action": "none"}
            else:
                # Determine severity based on the response
                severe_violations = ["threat", "hate", "abuse", "extremist", "scam", "phish", "drug", "violence"]
                is_severe = any(word in moderation_result.lower() for word in severe_violations)
                
                return {
                    "is_inappropriate": True,
                    "reason": moderation_result,
                    "action": "ban" if is_severe else "delete"
                }
        
        return {"is_inappropriate": False, "reason": None, "action": "none"}
    except Exception as e:
        logger.error(f"Moderation check failed: {e}")
        # If AI moderation fails, be conservative and check for basic profanity again
        if any(word in message_normalized for word in explicit_profanity):
            return {
                "is_inappropriate": True,
                "reason": "potentially inappropriate content",
                "action": "delete"
            }
        return {"is_inappropriate": False, "reason": None, "action": "none"}

def is_user_banned(room_id: str, fingerprint: str) -> bool:
    """Check if a user's fingerprint is banned in the room."""
    return room_id in banned_fingerprints and fingerprint in banned_fingerprints[room_id]

def ban_user(room_id: str, username: str, fingerprint: str):
    """Ban a user by their fingerprint and remove them from active users."""
    if room_id not in banned_fingerprints:
        banned_fingerprints[room_id] = set()
    banned_fingerprints[room_id].add(fingerprint)
    
    # Remove from active users if present
    if room_id in active_users:
        active_users[room_id].discard(username)
    
    # Log the fingerprint for this username
    if room_id not in user_fingerprints:
        user_fingerprints[room_id] = {}
    user_fingerprints[room_id][username] = fingerprint

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

# User Presence Management
@app.post("/api/chat/{room_id}/presence")
async def update_user_presence(room_id: str, payload: UserPresencePayload):
    """Updates user presence in a room (join/leave) with fingerprint checking."""
    if room_id not in chats:
        raise HTTPException(status_code=404, detail="Room not found")
    
    # Check if user is banned
    if is_user_banned(room_id, payload.fingerprint):
        raise HTTPException(status_code=403, detail="You have been banned from this chat room")
    
    if room_id not in active_users:
        active_users[room_id] = set()
    
    # Update fingerprint mapping
    if room_id not in user_fingerprints:
        user_fingerprints[room_id] = {}
    user_fingerprints[room_id][payload.username] = payload.fingerprint
    
    if payload.action == "join":
        active_users[room_id].add(payload.username)
        logger.info(f"User '{payload.username}' joined room {room_id}")
    elif payload.action == "leave":
        active_users[room_id].discard(payload.username)
        logger.info(f"User '{payload.username}' left room {room_id}")
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'join' or 'leave'")
    
    return {
        "active_users": list(active_users[room_id]),
        "count": len(active_users[room_id])
    }

@app.get("/api/chat/{room_id}/users")
async def get_active_users(room_id: str):
    """Returns the list of active users in a room."""
    if room_id not in chats:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {
        "active_users": list(active_users.get(room_id, set())),
        "count": len(active_users.get(room_id, set()))
    }

# Room Management
@app.post("/api/create_room")
async def create_room():
    """Creates a new chat room and returns its unique ID."""
    room_id = str(uuid4())[:8]  # Shorter, easier to share UUID
    while room_id in chats: # Ensure uniqueness, although collision is unlikely
        room_id = str(uuid4())[:8]
        
    chats[room_id] = []  # Initialize empty chat history
    active_users[room_id] = set()  # Initialize empty set of active users
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
async def post_message(room_id: str, payload: MessagePayload, request: Request):
    """Receives a user message, moderates it, adds to history if appropriate, gets AI response."""
    if room_id not in chats:
        raise HTTPException(status_code=404, detail="Room not found")
        
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="AI Service not configured by administrator.")

    username = payload.username if payload.username else "User"
    user_message_content = payload.message.strip()

    if not user_message_content:
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    # Check if user is banned by fingerprint
    if is_user_banned(room_id, payload.fingerprint):
        raise HTTPException(status_code=403, detail="You have been banned from this chat room")

    # Update fingerprint mapping
    if room_id not in user_fingerprints:
        user_fingerprints[room_id] = {}
    user_fingerprints[room_id][username] = payload.fingerprint

    # Moderate the message
    moderation_result = check_message_moderation(user_message_content, username)
    
    if moderation_result["is_inappropriate"]:
        if moderation_result["action"] == "ban":
            ban_user(room_id, username, payload.fingerprint)
            system_message = {
                "role": "system",
                "content": f"User '{username}' has been banned for inappropriate behavior: {moderation_result['reason']}",
                "username": "System",
                "timestamp": datetime.now().isoformat()
            }
            chats[room_id].append(system_message)
            raise HTTPException(status_code=403, detail=f"Banned: {moderation_result['reason']}")
        
        if moderation_result["action"] == "delete":
            deleted_message = {
                "role": "system",
                "content": f"[Message deleted by moderator - Reason: {moderation_result['reason']}]",
                "username": "System",
                "timestamp": datetime.now().isoformat()
            }
            chats[room_id].append(deleted_message)
            raise HTTPException(status_code=400, detail=f"Message deleted: {moderation_result['reason']}")

    timestamp = datetime.now().isoformat()

    # Add user message to history
    user_message_entry = {
        "role": "user",
        "content": user_message_content,
        "username": username,
        "timestamp": timestamp
    }
    chats[room_id].append(user_message_entry)
    logger.info(f"Room {room_id} - User '{username}': {user_message_content}")

    # Prepare messages for OpenRouter API
    api_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in chats[room_id]
        if msg.get("role") in ["user", "assistant"]
    ]

    # --- Refactored Streaming Logic with httpx --- 
    async def generate_response():
        full_content = ""
        assistant_timestamp = None
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                async with client.stream(
                    "POST",
                    url=OPENROUTER_API_URL,
                    headers=get_openrouter_headers(),
                    json={
                        "model": "openai/gpt-4o-mini",
                        "messages": api_messages,
                        "max_tokens": 1000,
                        "stream": True
                    }
                ) as response:
                    # Check for HTTP errors
                    response.raise_for_status()

                    # Initialize timestamp on first successful chunk received
                    assistant_timestamp = datetime.now().isoformat()

                    # Asynchronously iterate over the streamed lines
                    async for line in response.aiter_lines():
                        # Check for client disconnect
                        if await request.is_disconnected():
                            logger.warning(f"Client disconnected during stream for room {room_id}")
                            break # Exit the loop if client disconnected
                            
                        if line:
                            try:
                                json_str = line.removeprefix('data: ').strip()
                                if not json_str:
                                    continue
                                    
                                if json_str == '[DONE]':
                                    break
                                
                                chunk = json.loads(json_str)
                                if chunk.get('choices') and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        if content:
                                            full_content += content
                                            # Yield each chunk as a JSON event
                                            yield f"data: {json.dumps({'content': content, 'timestamp': assistant_timestamp})}

"
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON chunk: {line}")
                                continue
                            except Exception as e:
                                logger.error(f"Error processing stream line: {e} - Line: {line}")
                                continue # Try to continue with the next line

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during streaming request: {e.response.status_code} - {e.response.text}")
            error_msg = json.dumps({"error": f"AI service error: {e.response.status_code}"})
            yield f"data: {error_msg}

"
        except httpx.RequestError as e:
            logger.error(f"Request error during streaming: {e}")
            error_msg = json.dumps({"error": "Failed to connect to AI service"})
            yield f"data: {error_msg}

"
        except Exception as e:
            logger.error(f"Unexpected error during AI response generation: {e}", exc_info=True)
            error_msg = json.dumps({"error": "An unexpected error occurred"})
            yield f"data: {error_msg}

"
        finally:
             # Check if client is still connected before proceeding
            if not await request.is_disconnected():
                # Save the complete message to chat history if content was generated
                if full_content and assistant_timestamp:
                    assistant_message_entry = {
                        "role": "assistant",
                        "content": full_content,
                        "username": "AI Assistant",
                        "timestamp": assistant_timestamp
                    }
                    chats[room_id].append(assistant_message_entry)
                    logger.info(f"Room {room_id} - AI: {full_content[:80]}...")
                # Send a completion event even if there was an error or disconnect earlier
                # Client needs this to know the stream ended server-side
                yield f"data: [DONE]

"
            else:
                logger.info(f"Stream for room {room_id} ended due to client disconnect. Not saving final message.")

    # --- End Refactored Streaming Logic --- 

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream"
    )

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