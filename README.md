# Telegram Group Summarizer Bot (Using Gemini AI)

This Telegram bot provides on-demand summaries of recent conversations within a group chat using Google's Gemini AI API. It helps users quickly catch up on discussions without reading through potentially hundreds of messages.

## Features

* **Command Triggered Summarization:** Users initiate summarization via a specific command (e.g., `/summarize 50`).
* **Configurable Length:** Specify the number of recent messages to summarize (Default: 25, Max: 200).
* **AI-Powered Summaries:** Leverages the configured Google Gemini model (default: `gemini-1.5-flash`, configurable via `GEMINI_MODEL_NAME` env var) for generating summaries.
* **Message Referencing:** Summaries include message IDs (e.g., `(ID: 12345)`) to help users manually locate the original context within the chat history.
* **Group Chat Focus:** Designed specifically to operate within Telegram group and supergroup environments.
* **In-Memory Caching:** Temporarily stores recent messages (up to 500 per chat) in memory to allow for summarization based on messages received while the bot is active.
* **User Rate Limiting:** Includes a cooldown period (currently 60 seconds per user) for the `/summarize` command to prevent abuse and manage API load.
* **Enhanced Error Feedback:** Provides more specific feedback to users in case of issues like API errors, content filtering, or rate limits.

## Non-Goals

* Real-time or continuous, automatic summarization.
* Direct, clickable links from the summary to specific messages within private/standard group chats (Telegram API limitation).
* Advanced Natural Language Processing beyond summarization (e.g., question answering about the chat).
* Summarizing content of media files (images, videos, audio) beyond their text captions.
* Persistent storage of chat history across bot restarts (summaries are based *only* on the current in-memory cache).

## Setup Instructions

1.  **Prerequisites:**
    * Python 3.8+ installed.
    * A Telegram account.
    * Access to Google AI Studio and a generated Gemini API Key.

2.  **Create a Telegram Bot:**
    * Open Telegram and start a chat with [@BotFather](https://t.me/BotFather).
    * Use the `/newbot` command and follow the prompts to choose a name and username for your bot.
    * **Copy the generated API Token** - you'll need this.
    * **Crucially, disable Group Privacy mode:**
        * Send `/mybots` to BotFather.
        * Select your newly created bot.
        * Go to `Bot Settings` -> `Group Privacy`.
        * Click `Turn off`. This is **required** for the bot to receive messages in groups it's added to.

3.  **Get Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create an API key if you don't have one already.
    * **Copy the API Key**. Ensure the key is enabled for the Gemini API.

4.  **Prepare the Code:**
    * Clone this repository or download the code files (`bot.py`, `requirements.txt`, `README.md`). You may also want to create a `.env` file based on `.env.example` if provided, or create it manually.
    * Navigate to the project directory in your terminal.

5.  **Create Environment File (`.env`):**
    * Create a file named `.env` in the project directory.
    * Add the following lines, replacing the placeholders with your actual credentials:
        ```dotenv
        TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

        # Optional: Specify a different Gemini model (defaults to gemini-1.5-flash)
        # GEMINI_MODEL_NAME="gemini-1.5-pro"

        # Optional: Set your Telegram User ID here to receive error notifications from the bot
        # ADMIN_CHAT_ID="YOUR_TELEGRAM_USER_ID"
        ```
    * Find your Telegram User ID by talking to a bot like `@userinfobot`.

6.  **Install Dependencies:**
    * It's highly recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    * Install the exact dependencies listed in `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

7.  **Run the Bot:**
    ```bash
    python bot.py
    ```
    * If everything is configured correctly, you should see log messages indicating the bot has started polling. Keep this terminal window open or use a process manager (like `systemd`, `supervisor`, `screen`, or `tmux`) to run the bot persistently on a server.

## Usage

1.  **Add the Bot:** Invite the Telegram bot you created to your target group chat. Ensure it has permission to read messages (disabling group privacy via BotFather handles this).
2.  **Allow Caching:** The bot needs to observe messages being sent *after* it joins (and while it's running) to build its internal cache for summarization. Give it a minute or two in an active chat.
3.  **Request Summary:**
    * Type `/summarize` to summarize the default number of recent messages (currently 25).
    * Type `/summarize N` (e.g., `/summarize 100`) to summarize the last N messages from the cache (up to the maximum of 200).
4.  **View Summary:** The bot will send a "processing" message, then edit it to display the AI-generated summary, including message ID references. If you request summaries too quickly, it will ask you to wait due to the cooldown. If an error occurs, it will provide feedback.

## Important Considerations & Limitations

* **In-Memory Cache Only:** The bot only remembers messages sent *while it is running* and *up to its cache limit* (500 messages per chat). If the bot restarts, the cache is **cleared**, and it cannot summarize messages sent before it joined or while it was offline.
* **No Historical Fetching:** This bot *does not* fetch historical messages from Telegram upon request due to API limitations and design choices. It summarizes based *only* on the messages it has seen and cached since its last start.
* **Group Privacy:** As mentioned, Group Privacy **must** be disabled for the bot via BotFather for it to function in groups.
* **API Costs & Rate Limits:** Use of the Google Gemini API may incur costs based on usage and Google's pricing. Both Telegram and Google APIs have rate limits. The bot includes user-level cooldowns to help mitigate hitting these, but extremely high usage could still be an issue.
* **Summary Quality:** The quality and accuracy of the summary depend heavily on the Gemini model used, the quality of the prompt engineering in the code, and the clarity/content of the conversation itself.
* **Error Handling:** While error handling has been improved to provide better feedback, complex network issues or unexpected API changes might still cause disruptions. Check the bot's logs for detailed error information. If `ADMIN_CHAT_ID` is configured, critical errors will be forwarded via Telegram message.