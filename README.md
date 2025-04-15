# Group AI Chat

A simple, shareable group chat application where users in the same room interact with a shared AI assistant powered by the OpenRouter API.

![Screenshot Placeholder](https://via.placeholder.com/600x400.png?text=App+Screenshot)
*(Replace with an actual screenshot)*

## Features

-   **Shared Chat Rooms:** Create unique chat rooms via shareable links.
-   **Group AI Interaction:** All users in a room share the same conversation context with the AI (uses `openrouter/auto` model).
-   **Near Real-time Updates:** Uses polling to fetch new messages from other users and the AI automatically.
-   **Customizable Usernames:** Set your display name (saved locally in your browser).
-   **Theming:** Choose from various DaisyUI themes (saved locally).
-   **Simple Interface:** Minimalist design using TailwindCSS and DaisyUI.
-   **Typing Indicator:** Shows when the AI is processing a response.
-   **Timestamps:** Messages include timestamps.
-   **Copy Room Link:** Easily copy the room URL to share.
-   **Basic Error Handling:** Provides feedback for common issues (e.g., API errors, room not found).

## Tech Stack

-   **Backend:** Python 3.x, FastAPI
-   **Frontend:** HTML5, CSS3 (TailwindCSS + DaisyUI via CDN), Vanilla JavaScript
-   **AI Integration:** [OpenRouter.ai](https://openrouter.ai/) API (`openrouter/auto` model)
-   **Dependencies:** `requests`, `uvicorn`, `python-dotenv`

## Setup & Running Locally

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd group-ai-chat
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   Create a file named `.env` in the project root directory.
    *   Add your OpenRouter API key to it:
        ```dotenv
        OPENROUTER_API_KEY="your-actual-openrouter-api-key"
        # Optional: Customize for OpenRouter ranking/identification
        # YOUR_SITE_URL="http://localhost:8000"
        # YOUR_SITE_NAME="My AI Chat App"
        ```
    *   Get your API key from [OpenRouter.ai](https://openrouter.ai/).

5.  **Run the Backend Server:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Automatically restarts the server when code changes (for development).
    *   `--host 0.0.0.0`: Makes the server accessible on your local network. Use `127.0.0.1` to restrict access to only your machine.
    *   `--port 8000`: Specifies the port to run on.

6.  **Access the Application:**
    Open your web browser and navigate to `http://localhost:8000` (or the host/port you configured).

## How to Use

1.  **Homepage:** You'll land on the welcome screen.
2.  **Create Room:** Click "Create New Room". You'll be automatically redirected to a unique chat URL (e.g., `http://localhost:8000/chat/abcdefgh`).
3.  **Join Room:** Paste a Room ID (e.g., `abcdefgh`) into the input field and click "Join".
4.  **Share:** Copy the full room URL from your browser's address bar (or use the copy button in the navbar) and share it with others.
5.  **Chat:**
    *   Enter your desired username in the input field (it will be saved for your next visit).
    *   Type your message and press Enter or click "Send".
    *   Your message and messages from other users in the room will appear.
    *   The AI will respond based on the conversation history. New messages will appear automatically every few seconds (due to polling).

## Limitations & Potential Improvements

*   **Polling Inefficiency:** The current implementation fetches the *entire* chat history every few seconds. For very long chats or many users, this is inefficient. A better approach would be WebSockets or Server-Sent Events (SSE) for true real-time updates, or an API endpoint that returns only *new* messages since the last poll.
*   **In-Memory Storage:** Chat history is lost when the Python server restarts. Implement persistent storage (e.g., SQLite, PostgreSQL, Redis) for production use.
*   **No Authentication:** Anyone with the link can join a room. Add user accounts if privacy is needed.
*   **Basic Error Handling:** Could be made more robust and user-friendly.
*   **Scalability:** The single-file, in-memory approach won't scale well. Consider splitting the backend, using async database drivers, and containerization (Docker) for larger deployments.
*   **Context Window:** Long conversations might exceed the AI model's context window limit. Implement context truncation or summarization strategies.