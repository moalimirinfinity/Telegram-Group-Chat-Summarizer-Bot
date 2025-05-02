import logging
import os
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone

# Third-party libraries
import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Allow specifying model via env var, default to flash
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")


if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    # Log critical error during startup if Gemini config fails
    logging.critical(f"CRITICAL: Failed to configure Gemini AI: {e}", exc_info=True)
    raise ValueError(f"Failed to configure Gemini: {e}")


# Bot Configuration
COMMAND_NAME = "summarize"
DEFAULT_SUMMARY_MESSAGES = 25
MAX_SUMMARY_MESSAGES = 200  # Limit to prevent abuse/long processing
MESSAGE_CACHE_SIZE = 500    # Max messages to store per chat in memory

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx noise for cleaner logs
logger = logging.getLogger(__name__)

# --- In-Memory Message Cache ---
# Stores recent messages per chat_id
# Format: {chat_id: deque([(message_id, user_name, text, timestamp), ...])}
message_cache = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))

# --- Helper Functions ---

def format_message_for_gemini(msg_data: tuple) -> str:
    """Formats a single message tuple for the Gemini prompt."""
    message_id, user_name, text, timestamp = msg_data
    # Format timestamp for clarity in UTC
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
    # Basic escaping for user name to avoid markdown issues in LLM prompt context if needed
    safe_user_name = user_name.replace('[', '(').replace(']', ')')
    return f"[{ts_str} - {safe_user_name} (ID: {message_id})]: {text}"

def format_messages_for_prompt(messages: list[tuple]) -> str:
    """Formats a list of message tuples into a single string for the Gemini prompt."""
    return "\n".join(format_message_for_gemini(msg) for msg in messages)

# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start command."""
    await update.message.reply_text(
        "Hi! I'm a bot designed to summarize recent messages in this group.\n\n"
        f"Use the command `/{COMMAND_NAME} [N]` where `N` is the number of recent messages you want to summarize. "
        f"(Default: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}).\n"
        f"Example: `/{COMMAND_NAME} 50`\n\n"
        f"I use the `{GEMINI_MODEL_NAME}` model for generating summaries.\n\n"
        "**Important:** For me to see messages and summarize them, 'Group Privacy' mode must be **disabled** in my settings. "
        "You can manage this via @BotFather (`/mybots` -> select bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`).",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /summarize command to generate and send a summary."""
    message = update.message
    chat = message.chat
    chat_id = chat.id

    # Ensure the command is from a group chat where the bot should function
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        await message.reply_text("This command only works in group or supergroup chats.")
        return

    # Parse arguments to get the number of messages requested
    args = context.args
    try:
        num_messages = int(args[0]) if args else DEFAULT_SUMMARY_MESSAGES
        if not (0 < num_messages <= MAX_SUMMARY_MESSAGES):
            await message.reply_text(
                f"Please provide a number between 1 and {MAX_SUMMARY_MESSAGES}. "
                f"Usage: `/{COMMAND_NAME} [number]`",
                 parse_mode=constants.ParseMode.MARKDOWN
            )
            return
    except (ValueError, IndexError):
        await message.reply_text(
            f"Invalid number format. Usage: `/{COMMAND_NAME} [number]` (e.g., `/{COMMAND_NAME} 50`).",
             parse_mode=constants.ParseMode.MARKDOWN
        )
        return

    # Retrieve messages from the cache for this specific chat
    # Use .get() to safely handle cases where the chat_id might not be in the cache yet
    chat_messages_deque = message_cache.get(chat_id)
    if not chat_messages_deque:
        logger.info(f"No message cache found for chat {chat_id} when attempting summarization.")
        await message.reply_text(
            "I haven't stored any messages from this chat yet, or my cache was cleared. "
            "Please wait for more messages to arrive or ensure I have permission to read messages (Group Privacy off)."
        )
        return

    # Create a thread-safe copy of the relevant messages from the deque
    chat_messages = list(chat_messages_deque)

    # Check again if the list is empty after copying (should be redundant but safe)
    if not chat_messages:
        logger.info(f"Message cache for chat {chat_id} exists but is empty.")
        await message.reply_text(
            "My message cache for this chat is currently empty. Please wait for new messages."
        )
        return

    # Select the most recent N messages from the copied list
    messages_to_summarize = chat_messages[-num_messages:]
    actual_count = len(messages_to_summarize)

    logger.info(f"Attempting to summarize last {actual_count} messages for chat {chat_id} (requested: {num_messages}).")

    # Inform the user that processing is starting and store the message for later editing
    try:
        processing_msg = await message.reply_text(
            f"⏳ Fetching and summarizing the last {actual_count} cached messages using {GEMINI_MODEL_NAME}... please wait.",
            disable_notification=True # Be less noisy
        )
    except Exception as send_err:
        logger.error(f"Failed to send 'processing' message to chat {chat_id}: {send_err}")
        # If we can't even send the initial message, we probably can't send the result either.
        return

    # Format messages and construct the prompt for the Gemini API
    formatted_text = format_messages_for_prompt(messages_to_summarize)
    prompt = (
        "You are a helpful assistant tasked with summarizing Telegram group chat conversations.\n"
        "Provide a *concise, neutral, and objective* summary of the following messages. "
        "Focus on key discussion points, decisions made, questions asked, and any action items mentioned.\n"
        "Format the summary clearly, perhaps using bullet points.\n"
        "Where specific points are derived from messages, reference the original message ID(s) like '(ID: 123)' or '(IDs: 124, 125)'.\n\n"
        "--- Conversation Start ---\n"
        f"{formatted_text}\n"
        "--- Conversation End ---\n\n"
        "Concise Summary:"
    )

    try:
        # Run the potentially blocking Gemini API call in a separate thread
        generation_config = genai.types.GenerationConfig(
            # Optional: control randomness (e.g., lower temp for more factual summary)
            # temperature=0.7,
            # Optional: limit output length if needed
            # max_output_tokens=500
        )
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=generation_config,
            # Optional: Add safety settings if stricter filtering is desired
            # safety_settings={'HATE_SPEECH': 'BLOCK_ONLY_HIGH', ...}
        )

        # Validate the response from the Gemini API
        if not response or not hasattr(response, 'text') or not response.text:
             logger.warning(f"Gemini API returned an empty or invalid response for chat {chat_id}. Prompt length: {len(prompt)}. Response: {response}")
             raise ValueError("Received empty summary from API (possibly due to content filtering or API issue).")

        summary_text = response.text.strip() # Ensure leading/trailing whitespace is removed

        # Format the final reply for Telegram
        first_msg_id = messages_to_summarize[0][0]
        last_msg_id = messages_to_summarize[-1][0]

        reply_text = (
            f"✨ **Summary of last {actual_count} messages (approx. ID {first_msg_id} to {last_msg_id}):** ✨\n\n"
            f"{summary_text}\n\n"
            f"*Summary generated by {GEMINI_MODEL_NAME}. Note: Message IDs are for reference; you may need to scroll manually.*"
        )

        # Edit the original "processing" message with the final summary
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=processing_msg.message_id,
            text=reply_text,
            parse_mode=constants.ParseMode.MARKDOWN
        )
        logger.info(f"Successfully generated and sent summary for chat {chat_id} ({actual_count} messages).")

    except Exception as e:
        # Log the detailed error internally
        logger.error(f"Error during summary generation or sending for chat {chat_id}: {e}", exc_info=True)
        # Provide a generic, safe error message to the user by editing the status message
        user_error_message = "❌ Oops! Something went wrong while generating the summary. The administrators have been notified."
        try:
             await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg.message_id,
                text=user_error_message
             )
        except Exception as edit_err:
             # If editing fails, log it. Falling back to sending a new message might be too noisy.
             logger.error(f"Failed to edit message to display summary error for chat {chat_id}: {edit_err}")


# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stores incoming text messages from groups/supergroups in the cache."""
    message = update.message
    # Basic validation: ensure message object and text content exist
    if not message or not message.text:
        return

    chat = message.chat
    # Cache only messages from groups and supergroups where the bot might be asked to summarize
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        return

    chat_id = chat.id
    user = message.from_user
    # Ensure user info is available, provide a fallback name if not
    if not user:
        # This might happen for anonymous admin messages or channel posts forwarded to linked group
        user_name = "Anonymous"
    else:
        # Prefer first name, fallback to username, then to a generic user ID string
        user_name = user.first_name or user.username or f"User_{user.id}"

    message_id = message.message_id
    text = message.text
    # Ensure timestamp is timezone-aware UTC. Use current time as fallback if message.date is missing.
    timestamp = message.date if message.date else datetime.now(timezone.utc)
    if timestamp.tzinfo is None: # Ensure timezone aware (should be UTC from Telegram)
         timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Add message data tuple to this chat's cache deque
    try:
        message_cache[chat_id].append((message_id, user_name, text, timestamp))
        # Optional: Debug log to see caching in action
        # logger.debug(f"Cached message {message_id} from {user_name} in chat {chat_id}. Cache size: {len(message_cache[chat_id])}")
    except Exception as cache_err:
        logger.error(f"Failed to cache message {message_id} for chat {chat_id}: {cache_err}", exc_info=True)


# --- Error Handler ---

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    # Log the error and the update object type that caused it
    logger.error(f"Exception while handling an update {type(update)}:", exc_info=context.error)
    # Potentially add more sophisticated error handling here, e.g., notifying admins for specific errors


# --- Main Application Setup ---

def main() -> None:
    """Sets up the Application and starts the bot polling."""
    logger.info(f"--- Initializing Telegram Bot ---")
    logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    logger.info(f"Message cache size per chat: {MESSAGE_CACHE_SIZE}")
    logger.info(f"Default summary length: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}")

    try:
        # Create the Telegram Application builder
        application_builder = Application.builder().token(TELEGRAM_TOKEN)
        # Optional: configure connection pool size if needed for high load scenarios
        # application_builder.connection_pool_size(512)
        application = application_builder.build()

        # Register command handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", start_command)) # Alias help to start
        application.add_handler(CommandHandler(COMMAND_NAME, summarize_command))

        # Register message handler specifically for non-command text messages in groups/supergroups
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS,
            handle_message
        ))

        # Register the global error handler
        application.add_error_handler(error_handler)

        # Start polling for updates
        logger.info("Bot polling started...")
        # allowed_updates can be tuned to only process needed types, e.g., [Update.MESSAGE]
        application.run_polling(allowed_updates=Update.ALL_TYPES)

    except ValueError as e:
        # Handle critical configuration errors identified earlier
        logger.critical(f"Configuration Error: {e}")
        import sys
        sys.exit(1) # Exit if essential config (like tokens) is missing
    except Exception as e:
        logger.critical(f"Application failed to initialize or run polling: {e}", exc_info=True)
        import sys
        sys.exit(1) # Exit on other critical startup failures

    logger.info("--- Bot polling stopped ---")


if __name__ == "__main__":
    # Entry point for running the script
    main()