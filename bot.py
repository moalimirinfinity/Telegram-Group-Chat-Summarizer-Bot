import logging
import os
import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

# Third-party libraries
import google.generativeai as genai
# Import specific exceptions if the library provides them and they are useful
# from google.api_core.exceptions import GoogleAPIError # Example
from google.generativeai.types import GenerationConfig, SafetySettingDict, HarmCategory, HarmBlockThreshold
from google.generativeai.types.generation_types import BlockedPromptException, StopCandidateException # More specific exceptions
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.error import TelegramError # Import base Telegram error

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
    logger_gemini = logging.getLogger("GeminiAI") # Separate logger for Gemini interactions
    logger_gemini.info(f"Successfully configured Gemini AI with model: {GEMINI_MODEL_NAME}")
except Exception as e:
    # Log critical error during startup if Gemini config fails
    logging.critical(f"CRITICAL: Failed to configure Gemini AI: {e}", exc_info=True)
    # Exit cleanly instead of raising generic ValueError
    import sys
    print(f"CRITICAL: Failed to configure Gemini: {e}", file=sys.stderr)
    sys.exit(1)


# Bot Configuration
COMMAND_NAME = "summarize"
DEFAULT_SUMMARY_MESSAGES = 25
MAX_SUMMARY_MESSAGES = 200  # Limit to prevent abuse/long processing
MESSAGE_CACHE_SIZE = 500    # Max messages to store per chat in memory
SUMMARY_COOLDOWN_SECONDS = 60 # Cooldown per user for /summarize command

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Reduce httpx noise for cleaner logs unless debugging network issues
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- In-Memory Caches ---
# Stores recent messages per chat_id: {chat_id: deque([(message_id, user_name, text, timestamp), ...])}
message_cache = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))
# Stores last summary request time per user_id: {user_id: timestamp}
user_last_summary_request = {}

# --- Helper Functions ---

def sanitize_for_prompt(text: str) -> str:
    """
    Basic sanitization for text included in prompts.
    Currently focuses on user names. Message text is passed largely as-is.
    Further sanitization might be needed if prompt injection becomes an issue.
    """
    # Basic escaping for user name to avoid markdown issues in LLM prompt context
    return text.replace('[', '(').replace(']', ')')

def format_message_for_gemini(msg_data: tuple) -> str:
    """Formats a single message tuple for the Gemini prompt."""
    message_id, user_name, text, timestamp = msg_data
    # Format timestamp for clarity in UTC
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
    safe_user_name = sanitize_for_prompt(user_name)
    # Note: Passing user message text directly. Relying on LLM robustness / API safety filters.
    return f"[{ts_str} - {safe_user_name} (ID: {message_id})]: {text}"

def format_messages_for_prompt(messages: list[tuple]) -> str:
    """Formats a list of message tuples into a single string for the Gemini prompt."""
    return "\n".join(format_message_for_gemini(msg) for msg in messages)

async def edit_or_reply_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, processing_msg_id: int = None, parse_mode=None):
    """Tries to edit a message, falls back to sending a new one."""
    try:
        if processing_msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg_id,
                text=text,
                parse_mode=parse_mode
            )
        else:
            # If no processing message ID, just send a new message
             await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
    except TelegramError as e:
        logger.error(f"Failed to edit message {processing_msg_id} in chat {chat_id}: {e}. Attempting to send new message.")
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except TelegramError as send_err:
            logger.error(f"Failed even to send new message to chat {chat_id} after edit failed: {send_err}")


# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start or /help command."""
    await update.message.reply_text(
        "Hi! I'm a bot designed to summarize recent messages in this group.\n\n"
        f"Use the command `/{COMMAND_NAME} [N]` where `N` is the number of recent messages you want to summarize. "
        f"(Default: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}).\n"
        f"Example: `/{COMMAND_NAME} 50`\n\n"
        f"I use the `{GEMINI_MODEL_NAME}` model for generating summaries.\n"
        f"There's a {SUMMARY_COOLDOWN_SECONDS}-second cooldown per user for this command.\n\n"
        "**Important:** For me to see messages and summarize them, 'Group Privacy' mode must be **disabled** in my settings. "
        "You can manage this via @BotFather (`/mybots` -> select bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`).",
        parse_mode=constants.ParseMode.MARKDOWN
    )

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /summarize command to generate and send a summary."""
    message = update.message
    user = message.from_user
    chat = message.chat
    chat_id = chat.id
    user_id = user.id

    # 1. Check command origin
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        await message.reply_text("This command only works in group or supergroup chats.")
        return

    # 2. Rate Limiting Check
    now = time.monotonic() # Use monotonic clock for interval checks
    last_request_time = user_last_summary_request.get(user_id, 0)
    elapsed = now - last_request_time

    if elapsed < SUMMARY_COOLDOWN_SECONDS:
        remaining = round(SUMMARY_COOLDOWN_SECONDS - elapsed)
        await message.reply_text(
            f"⏳ Please wait {remaining} more seconds before requesting another summary.",
            disable_notification=True
        )
        return

    # 3. Parse arguments
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

    # 4. Retrieve messages from cache
    chat_messages_deque = message_cache.get(chat_id)
    if not chat_messages_deque:
        logger.info(f"No message cache found for chat {chat_id} when user {user_id} requested summarization.")
        await message.reply_text(
            "I haven't stored any messages from this chat yet, or my cache was cleared. "
            "Please wait for more messages to arrive or ensure I have permission to read messages (Group Privacy off)."
        )
        return

    # Create a thread-safe copy for processing
    chat_messages = list(chat_messages_deque)

    if not chat_messages:
        logger.info(f"Message cache for chat {chat_id} exists but is empty when user {user_id} requested summarization.")
        await message.reply_text(
            "My message cache for this chat is currently empty. Please wait for new messages."
        )
        return

    # Select the most recent N messages
    messages_to_summarize = chat_messages[-num_messages:]
    actual_count = len(messages_to_summarize)

    if actual_count == 0: # Should be covered by above checks, but good failsafe
        await message.reply_text("No messages found in the specified range to summarize.")
        return

    logger.info(f"User {user_id} requested summary of last {num_messages} messages in chat {chat_id}. Processing {actual_count} messages.")

    # 5. Send processing message
    processing_msg = None
    try:
        processing_msg = await message.reply_text(
            f"⏳ Fetching and summarizing the last {actual_count} cached messages using `{GEMINI_MODEL_NAME}`... please wait.",
            disable_notification=True,
            parse_mode=constants.ParseMode.MARKDOWN
        )
        processing_msg_id = processing_msg.message_id if processing_msg else None
    except TelegramError as send_err:
        logger.error(f"Failed to send 'processing' message to chat {chat_id} for user {user_id}: {send_err}")
        # If we can't send status, we probably can't send the result either. Abort early.
        return

    # 6. Format prompt and call Gemini API
    formatted_text = format_messages_for_prompt(messages_to_summarize)
    prompt = (
        "You are a helpful assistant tasked with summarizing Telegram group chat conversations.\n"
        "Provide a *concise, neutral, and objective* summary of the following messages. "
        "Focus on key discussion points, decisions made, questions asked, and any action items mentioned.\n"
        "Format the summary clearly, perhaps using bullet points or numbered lists.\n"
        "If specific points are derived from messages, reference the original message ID(s) like '(ID: 123)' or '(IDs: 124, 125)'.\n\n"
        "--- Conversation Start ---\n"
        f"{formatted_text}\n"
        "--- Conversation End ---\n\n"
        "Concise Summary:"
    )

    summary_text = ""
    error_occurred = False
    user_error_message = "❌ Oops! Something went wrong while generating the summary. Please try again later. If the problem persists, contact the bot admin." # Default error

    try:
        # Configure generation parameters
        generation_config = GenerationConfig(
            # temperature=0.7, # Optional: control creativity vs factualness
            # max_output_tokens=500 # Optional: limit summary length
        )
        # Configure safety settings (Example: block more strictly)
        safety_settings: SafetySettingDict = {
             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        # Run blocking API call in a separate thread
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # --- Refined Response Handling ---
        # Check for blocking first (most direct safety feedback)
        if not response.candidates:
             logger.warning(f"Gemini API returned no candidates for chat {chat_id}. Prompt length: {len(prompt)}. Feedback: {response.prompt_feedback}")
             # Try to get specific block reason
             block_reason = "Unknown reason (No candidates)"
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name
             logger.warning(f"Summary generation blocked for chat {chat_id}. Reason: {block_reason}")
             user_error_message = f"❌ Summary generation failed. The content may have triggered safety filters ({block_reason})."
             raise ValueError(f"Blocked prompt: {block_reason}") # Raise specific error for logging

        # Access text safely, checking candidate finish reason
        candidate = response.candidates[0]
        if candidate.finish_reason == StopCandidateException.FinishReason.SAFETY:
            logger.warning(f"Summary generation stopped due to safety concerns for chat {chat_id}. Finish Reason: SAFETY. Citations: {candidate.citation_metadata}")
            safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in candidate.safety_ratings])
            logger.warning(f"Safety Ratings: {safety_ratings_str}")
            user_error_message = "❌ Summary generation stopped due to safety concerns. The content might contain sensitive topics."
            raise StopCandidateException(f"Safety stop: {safety_ratings_str}") # Raise specific error
        elif candidate.finish_reason not in [StopCandidateException.FinishReason.STOP, StopCandidateException.FinishReason.MAX_TOKENS]:
             # Handle other non-standard finish reasons
             logger.warning(f"Summary generation finished unexpectedly for chat {chat_id}. Reason: {candidate.finish_reason.name}")
             user_error_message = f"❌ Summary generation finished unexpectedly ({candidate.finish_reason.name})."
             raise StopCandidateException(f"Unexpected finish: {candidate.finish_reason.name}")

        # Get the text content if valid
        if not candidate.content or not candidate.content.parts or not candidate.content.parts[0].text:
            logger.warning(f"Gemini API returned empty text content for chat {chat_id}. Finish Reason: {candidate.finish_reason.name}. Response: {response}")
            user_error_message = "❌ Summary generation resulted in empty content. Please try again later."
            raise ValueError("Received empty text part from API.")

        summary_text = candidate.content.parts[0].text.strip()

        # Check again if summary is empty after stripping
        if not summary_text:
             logger.warning(f"Gemini API returned whitespace-only summary for chat {chat_id}.")
             user_error_message = "❌ Summary generation resulted in an empty summary."
             raise ValueError("Received whitespace-only summary.")

        # If successful, update the rate limit timestamp *after* the successful API call
        user_last_summary_request[user_id] = time.monotonic()

    # --- Refined Exception Handling ---
    except BlockedPromptException as e:
        # Already logged reason, user message set above
        logger.error(f"BlockedPromptException during summary generation for chat {chat_id}: {e}", exc_info=False) # Log less verbosely
        error_occurred = True
        # user_error_message is already set
    except StopCandidateException as e:
        # Specific stops (Safety, other unexpected) logged, user message set above
        logger.error(f"StopCandidateException during summary generation for chat {chat_id}: {e}", exc_info=False) # Log less verbosely
        error_occurred = True
        # user_error_message is already set
    except Exception as e:
        # Catch other potential API errors or logic errors
        logger.error(f"Generic error during summary generation or processing for chat {chat_id}: {e}", exc_info=True)
        # Provide a more generic error message for unexpected issues
        user_error_message = "❌ An unexpected error occurred while generating the summary. The administrators have been notified."
        error_occurred = True

    # 7. Send Result or Error Message
    if not error_occurred and summary_text:
        first_msg_id = messages_to_summarize[0][0]
        last_msg_id = messages_to_summarize[-1][0]
        reply_text = (
            f"✨ **Summary of last {actual_count} messages (approx. ID {first_msg_id} to {last_msg_id}):** ✨\n\n"
            f"{summary_text}\n\n"
            f"*Summary generated by `{GEMINI_MODEL_NAME}`. Message IDs are for reference.*"
        )

        await edit_or_reply_message(context, chat_id, reply_text, processing_msg_id, constants.ParseMode.MARKDOWN)
        logger.info(f"Successfully generated and sent summary for chat {chat_id} ({actual_count} messages) requested by user {user_id}.")

    else:
        # An error occurred, use the specific or generic user_error_message
        await edit_or_reply_message(context, chat_id, user_error_message, processing_msg_id)
        logger.warning(f"Failed to generate summary for chat {chat_id} requested by user {user_id}. Sent error message to user.")


# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stores incoming text messages from groups/supergroups in the cache."""
    message = update.message
    # Basic validation: ensure message object and text content exist
    if not message or not message.text or message.via_bot: # Ignore messages via bots (including self)
        return

    chat = message.chat
    # Cache only messages from groups and supergroups
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        return

    chat_id = chat.id
    user = message.from_user

    # Ensure user info is available, provide a fallback name if not
    if not user:
        user_name = "Anonymous" # e.g., anonymous admin or channel post
    else:
        # Prefer first name, fallback to username, then to a generic user ID string
        user_name = user.first_name or user.username or f"User_{user.id}"

    message_id = message.message_id
    text = message.text
    # Ensure timestamp is timezone-aware UTC. Use current time as fallback if message.date is missing.
    timestamp = message.date if message.date else datetime.now(timezone.utc)
    if timestamp.tzinfo is None: # Ensure timezone aware
         timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Add message data tuple to this chat's cache deque
    try:
        # Use a tuple for immutability, good practice for data storage
        message_data = (message_id, user_name, text, timestamp)
        message_cache[chat_id].append(message_data)
        # Optional: Debug log to see caching in action
        # logger.debug(f"Cached message {message_id} from '{user_name}' in chat {chat_id}. Cache size: {len(message_cache[chat_id])}")
    except Exception as cache_err:
        logger.error(f"Failed to cache message {message_id} for chat {chat_id}: {cache_err}", exc_info=True)


# --- Error Handler ---

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Exception while handling an update:", exc_info=context.error)

    # Optionally add more specific error handling/reporting here if needed
    # For example, notify an admin user for critical errors:
    # if isinstance(context.error, SomeCriticalError):
    #    admin_chat_id = os.getenv("ADMIN_CHAT_ID")
    #    if admin_chat_id:
    #       await context.bot.send_message(chat_id=admin_chat_id, text=f"Critical error: {context.error}")


# --- Main Application Setup ---

def main() -> None:
    """Sets up the Application and starts the bot polling."""
    logger.info(f"--- Initializing Telegram Bot (PID: {os.getpid()}) ---")
    logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    logger.info(f"Message cache size per chat: {MESSAGE_CACHE_SIZE}")
    logger.info(f"Default summary length: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}")
    logger.info(f"Summary command cooldown: {SUMMARY_COOLDOWN_SECONDS} seconds per user")

    # Create the Telegram Application
    # Consider adding persistence if needed: e.g. PicklePersistence(filepath='bot_data.pkl')
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", start_command)) # Alias help to start
    application.add_handler(CommandHandler(COMMAND_NAME, summarize_command))

    # Register message handler specifically for non-command text messages in groups/supergroups
    # Ensure it only captures TEXT messages and excludes commands and messages via bots.
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS & ~filters.VIA_BOT,
        handle_message
    ))

    # Register the global error handler
    application.add_error_handler(error_handler)

    # Start polling for updates
    logger.info("Bot polling started...")
    # allowed_updates can be tuned for efficiency if only messages are needed
    # application.run_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY]) # Example
    application.run_polling(allowed_updates=Update.ALL_TYPES) # Default: process all update types

    logger.info("--- Bot polling stopped ---")


if __name__ == "__main__":
    main()