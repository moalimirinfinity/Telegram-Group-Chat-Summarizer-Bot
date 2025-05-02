# Telegram Group Summarizer Bot (Using Gemini AI)

This Telegram bot provides on-demand summaries of recent conversations within a group chat using Google's Gemini AI API. It helps users quickly catch up on discussions without reading through potentially hundreds of messages.

## Features

* **Command Triggered Summarization:** Users initiate summarization via a specific command (e.g., `/summarize 50`).
* **Configurable Length:** Specify the number of recent messages to summarize.
* **AI-Powered Summaries:** Leverages the configured Google Gemini model (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`) for generating summaries.
* **Message Referencing:** Summaries include message IDs (e.g., `(ID: 12345)`) to help users manually locate the original context within the chat history.
* **Group Chat Focus:** Designed specifically to operate within Telegram group and supergroup environments.
* **In-Memory Caching:** Temporarily stores recent messages in memory to allow for summarization based on messages received while the bot is active.

## Non-Goals

* Real-time or continuous, automatic summarization.
* Direct, clickable links from the summary to specific messages within private/standard group chats (Telegram API limitation).
* Advanced Natural Language Processing beyond summarization (e.g., question answering about the chat).
* Summarizing content of media files (images, videos, audio) beyond their text captions.
* Persistent storage of chat history across bot restarts (summaries are based on the current in-memory cache).

## Setup Instructions

1.  **Prerequisites:**
    * Python 3.8+ installed.
    * A Telegram account.
    * Access to Google AI Studio and a generated Gemini API Key.

2.  **Create a Telegram Bot:**
    * Open Telegram and start a chat with [@BotFather](https://t.me/BotFather).
    * Use the `/newbot` command and follow the prompts to choose a name and username for your bot.
    * **Copy the generated API Token** - you'll need this shortly.
    * **Crucially, disable Group Privacy mode:**
        * Send `/mybots` to BotFather.
        * Select your newly created bot.
        * Go to `Bot Settings` -> `Group Privacy`.
        * Click `Turn off`. This is **required** for the bot to receive messages in groups.

3.  **Get Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create an API key if you don't have one already.
    * **Copy the API Key**.

4.  **Prepare the Code:**
    * Clone this repository or download the code files (`bot.py`, `requirements.txt`, `README.md`, `.env.example`).
    * Navigate to the project directory in your terminal.

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Configure Environment Variables:**
    * Rename the `.env.example` file to `.env`.
    * Open the `.env` file and replace the placeholder values with your actual **Telegram Bot Token** and **Gemini API Key**.
    * (Optional) You can uncomment and set `GEMINI_MODEL_NAME` if you want to use a specific model like `gemini-1.5-pro` instead of the default `gemini-1.5-flash`.

7.  **Run the Bot:**
    ```bash
    python bot.py
    ```
    If everything is configured correctly, you should see log messages indicating the bot has started polling. Keep this terminal window open or use a process manager (like `systemd`, `supervisor`, `screen`, or `tmux`) to run the bot persistently on a server.

## Usage

1.  **Add the Bot:** Invite the Telegram bot you created to your target group chat. Ensure it has permission to read messages (disabling group privacy via BotFather handles this).
2.  **Allow Caching:** The bot needs to observe messages being sent *after* it joins (and while it's running) to build its internal cache for summarization.
3.  **Request Summary:**
    * Type `/summarize` to summarize the default number of recent messages (currently {DEFAULT_SUMMARY_MESSAGES}).
    * Type `/summarize N` (e.g., `/summarize 100`) to summarize the last N messages from the cache (up to the configured maximum {MAX_SUMMARY_MESSAGES}).
4.  **View Summary:** The bot will send a "processing" message, then edit it to display the AI-generated summary, including message ID references.

## Important Considerations & Limitations

* **In-Memory Cache Only:** The bot only remembers messages sent *while it is running* and *up to its cache limit* ({MESSAGE_CACHE_SIZE} messages per chat). If the bot restarts, the cache is **cleared**, and it cannot summarize messages sent before it joined or while it was offline.
* **No Historical Fetching:** This bot *does not* fetch historical messages from Telegram upon request due to API limitations. It summarizes based *only* on the messages it has seen and cached.
* **Group Privacy:** As mentioned, Group Privacy **must** be disabled for the bot via BotFather for it to function.
* **API Costs & Rate Limits:** Use of the Google Gemini API may incur costs based on usage and Google's pricing. Both Telegram and Google APIs have rate limits; excessive summarization requests in very active groups could potentially hit these limits.
* **Summary Quality:** The quality and accuracy of the summary depend on the Gemini model, the prompt used, and the content of the conversation.
* **Error Handling:** While basic error handling is included, complex network issues or API errors might still cause disruptions. Check the bot's logs for detailed error information.