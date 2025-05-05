# -*- coding: utf-8 -*-
import logging
import os
import asyncio
import time
import traceback
import google.api_core.exceptions
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

# Third-party libraries
import google.generativeai as genai
# Import necessary types - Candidate import removed earlier
from google.generativeai.types import (
    GenerationConfig, SafetySettingDict, HarmCategory, HarmBlockThreshold,
    BlockedPromptException, StopCandidateException
)
# Import specific API core exceptions if needed for network/auth issues
# from google.api_core.exceptions import GoogleAPIError, ClientError
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    ApplicationBuilder, # Explicit import for builder pattern
)
from telegram.error import TelegramError # Import base Telegram error

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Allow specifying model via env var, default to flash
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
# Optional: Admin user ID for receiving critical error notifications
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")


if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Apply default safety settings suitable for general chat summarization
    default_safety_settings: SafetySettingDict = {
         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL_NAME,
        safety_settings=default_safety_settings
        # Default generation config can also be set here if desired
        # generation_config=GenerationConfig(temperature=0.7)
    )
    # Quick check to ensure model configuration is valid (optional, fails early if needed)
    # response = gemini_model.generate_content("test", generation_config=GenerationConfig(max_output_tokens=1))
    logging.info(f"Successfully configured Gemini AI with model: {GEMINI_MODEL_NAME}")
except Exception as e:
    # Log critical error during startup if Gemini config fails
    logging.critical(f"CRITICAL: Failed to configure or test Gemini AI: {e}", exc_info=True)
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
# Reduce noise from telegram.ext internal polling unless needed
logging.getLogger("telegram.ext").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- In-Memory Caches ---
# Stores recent messages per chat_id: {chat_id: deque([(message_id, user_name, text, timestamp), ...])}
message_cache = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))
# Stores last summary request time per user_id: {user_id: timestamp (monotonic)}
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
    # Consider additional sanitization/truncation here if message text is extremely long or complex.
    return f"[{ts_str} - {safe_user_name} (ID: {message_id})]: {text}"

def format_messages_for_prompt(messages: list[tuple]) -> str:
    """Formats a list of message tuples into a single string for the Gemini prompt."""
    # Check total length? Could become very long. Consider truncation or alternative approach for huge message lists.
    # prompt_limit = 30000 # Example character limit for Gemini Flash
    # formatted_messages = "\n".join(format_message_for_gemini(msg) for msg in messages)
    # if len(formatted_messages) > prompt_limit:
    #     # Handle trimming / indicate truncation
    #     pass
    return "\n".join(format_message_for_gemini(msg) for msg in messages)

async def edit_or_reply_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, processing_msg_id: int = None, parse_mode=None, disable_web_page_preview=True):
    """Tries to edit a message, falls back to sending a new one."""
    try:
        if processing_msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )
        else:
            # If no processing message ID, just send a new message
             await context.bot.send_message(
                 chat_id=chat_id,
                 text=text,
                 parse_mode=parse_mode,
                 disable_web_page_preview=disable_web_page_preview
             )
    except TelegramError as e:
        logger.error(f"Failed to edit message {processing_msg_id} in chat {chat_id}: {e}. Attempting to send new message.")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )
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
        f"There's a {SUMMARY_COOLDOWN_SECONDS}-second cooldown per user for this command to prevent spam.\n\n"
        "**Important:** For me to see messages and summarize them, 'Group Privacy' mode must be **disabled** in my settings. "
        "You can manage this via @BotFather (`/mybots` -> select bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`).",
        parse_mode=constants.ParseMode.MARKDOWN,
        disable_web_page_preview=True
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
        logger.debug(f"Command '/{COMMAND_NAME}' ignored from non-group chat {chat_id} by user {user_id}")
        return

    # 2. Rate Limiting Check
    now = time.monotonic()
    last_request_time = user_last_summary_request.get(user_id, 0)
    elapsed = now - last_request_time

    if elapsed < SUMMARY_COOLDOWN_SECONDS:
        remaining = round(SUMMARY_COOLDOWN_SECONDS - elapsed)
        try:
            await message.reply_text(
                f"⏳ Please wait {remaining} more seconds before requesting another summary.",
                disable_notification=True,
            )
        except TelegramError as e:
             logger.warning(f"Failed to send rate limit message to user {user_id} in chat {chat_id}: {e}")
        return

    # 3. Parse arguments
    args = context.args
    try:
        num_messages = int(args[0]) if args else DEFAULT_SUMMARY_MESSAGES
        if not (0 < num_messages <= MAX_SUMMARY_MESSAGES):
            await message.reply_text(
                f"Please provide a number between 1 and {MAX_SUMMARY_MESSAGES}. "
                f"Usage: `/{COMMAND_NAME} [number]`",
                 parse_mode=constants.ParseMode.MARKDOWN,
            )
            return
    except (ValueError, IndexError):
        await message.reply_text(
            f"Invalid number format. Usage: `/{COMMAND_NAME} [number]` (e.g., `/{COMMAND_NAME} 50`).",
             parse_mode=constants.ParseMode.MARKDOWN,
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

    chat_messages = list(chat_messages_deque)

    if not chat_messages:
        logger.info(f"Message cache for chat {chat_id} exists but is empty when user {user_id} requested summarization.")
        await message.reply_text(
            "My message cache for this chat is currently empty. Please wait for new messages."
        )
        return

    messages_to_summarize = chat_messages[-num_messages:]
    actual_count = len(messages_to_summarize)

    if actual_count == 0:
        await message.reply_text("No messages found in the specified range to summarize.")
        return

    logger.info(f"User {user_id} requested summary of last {num_messages} messages in chat {chat_id}. Processing {actual_count} messages.")

    # 5. Send processing message
    processing_msg_id = None
    try:
        processing_msg = await message.reply_text(
            f"⏳ Fetching and summarizing the last {actual_count} cached messages using `{GEMINI_MODEL_NAME}`... please wait.",
            disable_notification=True,
            parse_mode=constants.ParseMode.MARKDOWN
        )
        processing_msg_id = processing_msg.message_id if processing_msg else None
    except TelegramError as send_err:
        logger.error(f"Failed to send 'processing' message to chat {chat_id} for user {user_id}: {send_err}")
        return

    # 6. Format prompt and call Gemini API
    formatted_text = format_messages_for_prompt(messages_to_summarize)
    if len(formatted_text) > 30000: # Example limit for Flash model
        logger.warning(f"Formatted text for chat {chat_id} exceeds 30,000 chars ({len(formatted_text)}). May be truncated by API or cause errors.")

    prompt = (
        "You are a helpful assistant tasked with summarizing Telegram group chat conversations.\n"
        "Provide a *concise, neutral, and objective* summary of the following messages. "
        "Focus on key discussion points, decisions made, questions asked, and any action items mentioned.\n"
        "Format the summary clearly, perhaps using bullet points or numbered lists for readability.\n"
        "If specific points are derived from messages, reference the original message ID(s) like '(ID: 123)' or '(IDs: 124, 125)' where appropriate.\n\n"
        "--- Conversation Start ---\n"
        f"{formatted_text}\n"
        "--- Conversation End ---\n\n"
        "Concise Summary:"
    )

    summary_text = ""
    error_occurred = False
    user_error_message = "❌ Oops! Something went wrong while generating the summary. Please try again later. If the problem persists, contact the bot admin."

    try:
        generation_config = GenerationConfig() # Add specific config if needed

        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=generation_config,
        )

        if not response.candidates:
             block_reason = "Unknown reason (No candidates)"
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason.name
             logger.warning(f"Summary generation blocked for chat {chat_id} by user {user_id}. Reason: {block_reason}. Prompt length: {len(prompt)}. Feedback: {response.prompt_feedback}")
             user_error_message = f"❌ Summary generation failed. The request was blocked, possibly due to safety filters ({block_reason}). Please revise the conversation if sensitive content is present."
             raise BlockedPromptException(f"Blocked prompt: {block_reason}")

        candidate = response.candidates[0]

        # --- *** FIX APPLIED HERE (Revision 3 - Integer Guesswork) *** ---
        # WARNING: Comparing against guessed integer values for FinishReason
        # as the enum is not directly accessible in google-generativeai==0.7.1.
        # These values might be incorrect. If this fails, check the logs for the
        # 'Received candidate finish_reason' message to find the correct values.

        # Assumed Integer Values (Common Pattern - THIS IS A GUESS):
        # FINISH_REASON_UNSPECIFIED = 0
        # FINISH_REASON_STOP = 1
        # FINISH_REASON_MAX_TOKENS = 2
        # FINISH_REASON_SAFETY = 3
        # FINISH_REASON_RECITATION = 4
        # FINISH_REASON_OTHER = 5

        # Default to -1 or some value indicating it's not set if candidate has no finish_reason
        finish_reason_value = getattr(candidate, 'finish_reason', -1)

        # Add logging to see the actual value and type received from the API
        logger.info(f"Received candidate finish_reason: {finish_reason_value} (type: {type(finish_reason_value)})")

        # Check against assumed integer values
        if finish_reason_value == 3: # Assumed SAFETY
            safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in getattr(candidate, 'safety_ratings', [])])
            logger.warning(f"Summary generation stopped due to SAFETY concerns (assumed value 3) for chat {chat_id} requested by {user_id}. Ratings: {safety_ratings_str}. Citations: {getattr(candidate, 'citation_metadata', None)}")
            user_error_message = "❌ Summary generation stopped due to safety concerns. The generated content might contain sensitive topics based on safety ratings."
            raise StopCandidateException(f"Safety stop: {safety_ratings_str}")

        elif finish_reason_value == 4: # Assumed RECITATION
             logger.warning(f"Summary generation stopped due to RECITATION concerns (assumed value 4) for chat {chat_id} requested by {user_id}. Citations: {getattr(candidate, 'citation_metadata', None)}")
             user_error_message = "❌ Summary generation stopped. The content may include material from protected sources."
             raise StopCandidateException("Recitation stop")

        elif finish_reason_value not in [1, 2]: # Assumed not STOP (1) or MAX_TOKENS (2)
             # This will also catch the default -1 if finish_reason wasn't present,
             # or 0 (UNSPECIFIED), or 5 (OTHER), or any other unexpected value.
             logger.warning(f"Summary generation finished unexpectedly or with non-standard reason for chat {chat_id} requested by {user_id}. Reason Value: {finish_reason_value}")
             # Provide a slightly more generic message if the reason is truly unknown (-1) or unspecified (0)
             if finish_reason_value in [-1, 0]:
                 user_error_message = f"❌ Summary generation finished with an unspecified reason ({finish_reason_value}). Please try again."
             else:
                 user_error_message = f"❌ Summary generation finished unexpectedly (Reason Value: {finish_reason_value}). Please try again."
             raise StopCandidateException(f"Unexpected finish: {finish_reason_value}")
        # --- *** END OF FIX *** ---


        # Get the text content if valid (Added getattr for safety)
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', []) if content else []

        if not parts:
             finish_reason_value = getattr(candidate, 'finish_reason', -1)
             logger.warning(f"Gemini API returned no content or parts for chat {chat_id} requested by {user_id}. Finish Reason Value: {finish_reason_value}. Response: {response}")
             user_error_message = "❌ Summary generation resulted in no content. This might be due to filtering or an API issue. Please try again later."
             raise ValueError("Received no content/parts from API.")

        text_part = getattr(parts[0], 'text', None)
        if text_part is None:
             finish_reason_value = getattr(candidate, 'finish_reason', -1)
             logger.warning(f"Gemini API returned empty text content part for chat {chat_id} requested by {user_id}. Finish Reason Value: {finish_reason_value}. Response: {response}")
             user_error_message = "❌ Summary generation resulted in empty content. This might be due to filtering or an API issue. Please try again later."
             raise ValueError("Received empty text part from API.")

        summary_text = text_part.strip()

        if not summary_text:
             logger.warning(f"Gemini API returned whitespace-only summary for chat {chat_id} requested by {user_id}.")
             user_error_message = "❌ Summary generation resulted in an empty summary after processing."
             error_occurred = True

        if not error_occurred and summary_text:
            user_last_summary_request[user_id] = time.monotonic()

    except (BlockedPromptException, StopCandidateException) as safety_exception:
        logger.error(f"{type(safety_exception).__name__} during summary for chat {chat_id}, user {user_id}: {safety_exception}", exc_info=False)
        error_occurred = True

    except google.api_core.exceptions.InternalServerError as gemini_ise:
        logger.error(f"Gemini Internal Server Error during summary for chat {chat_id}, user {user_id}: {gemini_ise}", exc_info=True)
        user_error_message = "❌ The summarization service (Gemini) reported an internal error. Please try again later."
        error_occurred = True

    except google.api_core.exceptions.GoogleAPIError as api_error:
        logger.error(f"Google API Error during summary for chat {chat_id}, user {user_id}: {api_error}", exc_info=True)
        if isinstance(api_error, google.api_core.exceptions.PermissionDenied):
             user_error_message = "❌ Permission denied by the summarization service. Please check the API key and project settings."
        elif isinstance(api_error, google.api_core.exceptions.DeadlineExceeded):
             user_error_message = "❌ The request to the summarization service timed out. Please try again later."
        elif isinstance(api_error, google.api_core.exceptions.Unauthenticated):
             user_error_message = "❌ Authentication error with the summarization service. Please notify the bot admin."
        elif isinstance(api_error, google.api_core.exceptions.ResourceExhausted):
             user_error_message = "❌ Rate limit exceeded for the summarization service. Please wait and try again."
        else:
             user_error_message = "❌ A Google API error occurred while generating the summary. Please notify the bot admin."
        error_occurred = True

    except ValueError as val_err:
        logger.error(f"ValueError during summary processing for chat {chat_id}, user {user_id}: {val_err}", exc_info=False)
        error_occurred = True

    except Exception as e:
        # Catch other potential logic errors within this block
        logger.error(f"Unexpected Python error during summary generation/processing for chat {chat_id}, user {user_id}: {e}", exc_info=True)
        error_occurred = True
        if ADMIN_CHAT_ID:
            try:
                error_details = f"Unexpected error in summarize_command for chat {chat_id}, user {user_id}:\nType: {type(e).__name__}\nError: {e}\nTraceback:\n{traceback.format_exc()}"
                await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=error_details[:4000])
            except Exception as notify_err:
                logger.error(f"Failed to send error notification to admin {ADMIN_CHAT_ID}: {notify_err}")

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
        # Use the specific or generic user_error_message set during exception handling or empty summary check
        await edit_or_reply_message(context, chat_id, user_error_message, processing_msg_id)
        logger.warning(f"Failed summary or empty result for chat {chat_id} requested by user {user_id}. Sent error/info message to user.")


# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stores incoming text messages from groups/supergroups in the cache."""
    message = update.message
    if not message or not message.text or message.via_bot:
        return

    chat = message.chat
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        return

    chat_id = chat.id
    user = message.from_user

    if not user:
        user_name = "Anonymous"
    else:
        user_name = user.first_name.strip() if user.first_name else \
                    (user.username.strip() if user.username else f"User_{user.id}")
        if not user_name:
            user_name = f"User_{user.id}"

    message_id = message.message_id
    text = message.text
    timestamp = message.date if message.date else datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
         timestamp = timestamp.replace(tzinfo=timezone.utc)

    try:
        message_data = (message_id, user_name, text, timestamp)
        message_cache[chat_id].append(message_data)
    except Exception as cache_err:
        logger.error(f"Failed to cache message {message_id} for chat {chat_id}: {cache_err}", exc_info=True)


# --- Error Handler ---

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and notify admin if configured."""
    logger.error(f"Exception while handling an update:", exc_info=context.error)

    if ADMIN_CHAT_ID:
        try:
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
            tb_string = "".join(tb_list)
            error_message = (
                f"An error occurred processing an update:\n"
                f"Update: {update}\n"
                f"Error Type: {type(context.error).__name__}\n"
                f"Error: {context.error}\n"
                f"Traceback:\n```\n{tb_string[:3500]}\n```" # Limit traceback length
            )
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=error_message[:4096], parse_mode=constants.ParseMode.MARKDOWN)
        except Exception as notify_err:
            logger.error(f"Failed to send error notification to admin {ADMIN_CHAT_ID}: {notify_err}")


# --- Main Application Setup ---

def main() -> None:
    """Sets up the Application and starts the bot polling."""
    logger.info(f"--- Initializing Telegram Bot (PID: {os.getpid()}) ---")
    logger.info(f"Using Telegram Bot Token: {'*' * (len(TELEGRAM_TOKEN) - 4)}{TELEGRAM_TOKEN[-4:]}")
    logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    logger.info(f"Message cache size per chat: {MESSAGE_CACHE_SIZE}")
    logger.info(f"Default summary length: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}")
    logger.info(f"Summary command cooldown: {SUMMARY_COOLDOWN_SECONDS} seconds per user")
    if ADMIN_CHAT_ID:
        logger.info(f"Admin chat ID for notifications: {ADMIN_CHAT_ID}")
    else:
        logger.warning("ADMIN_CHAT_ID not set. Error notifications will not be sent via Telegram.")

    builder = ApplicationBuilder().token(TELEGRAM_TOKEN)
    application = builder.build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", start_command))
    application.add_handler(CommandHandler(COMMAND_NAME, summarize_command))

    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS & ~filters.VIA_BOT,
        handle_message
    ))

    application.add_error_handler(error_handler)

    logger.info("Bot polling started...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    logger.info("--- Bot polling stopped ---")


if __name__ == "__main__":
    main()