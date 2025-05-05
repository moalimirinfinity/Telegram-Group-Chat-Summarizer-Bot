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
# Import necessary types for v0.8.5+
# Candidate class itself might not be needed for direct import if accessing via response
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
    # Using BLOCK_MEDIUM_AND_ABOVE as a reasonable default
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
    # Quick check removed as it might consume quota unnecessarily during startup
    logging.info(f"Successfully configured Gemini AI with model: {GEMINI_MODEL_NAME}")
except Exception as e:
    logging.critical(f"CRITICAL: Failed to configure or test Gemini AI: {e}", exc_info=True)
    import sys
    print(f"CRITICAL: Failed to configure Gemini: {e}", file=sys.stderr)
    sys.exit(1) # Exit if core AI configuration fails

# Bot Configuration
COMMAND_NAME = "summarize"
DEFAULT_SUMMARY_MESSAGES = 25
MAX_SUMMARY_MESSAGES = 200  # Limit to prevent abuse/long processing
MESSAGE_CACHE_SIZE = 500    # Max messages to store per chat in memory
SUMMARY_COOLDOWN_SECONDS = 60 # Cooldown per user for /summarize command

# --- Logging Setup ---
# Use ISO format for timestamps
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%dT%H:%M:%S%z' # ISO 8601 format
)
# Filter out excessive noise from underlying libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.INFO) # Keep INFO for connection status etc.
logger = logging.getLogger(__name__)

# --- In-Memory Caches ---
# Stores recent messages per chat_id: {chat_id: deque([(message_id, user_name, text, timestamp), ...])}
message_cache: dict[int, deque] = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))
# Stores last summary request time per user_id: {user_id: timestamp (monotonic)}
user_last_summary_request: dict[int, float] = {}

# --- Helper Functions ---

def sanitize_for_prompt(text: str) -> str:
    """Basic sanitization for user names included in prompts."""
    # Simple replacement to avoid markdown issues or potential injection points in names
    return text.replace('[', '(').replace(']', ')')

def format_message_for_gemini(msg_data: tuple) -> str:
    """Formats a single message tuple for the Gemini prompt."""
    message_id, user_name, text, timestamp = msg_data
    # Format timestamp in ISO 8601 UTC for clarity
    ts_str = timestamp.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    safe_user_name = sanitize_for_prompt(user_name)
    # Ensure text is included, even if empty, to maintain structure
    text_content = text if text is not None else ""
    return f"[{ts_str} - {safe_user_name} (ID: {message_id})]: {text_content}"

def format_messages_for_prompt(messages: list[tuple]) -> str:
    """Formats a list of message tuples into a single string for the Gemini prompt."""
    # Consider adding a check or warning if the formatted string becomes excessively long
    # (e.g., > 30k characters for Flash models) although the API might handle truncation.
    return "\n".join(format_message_for_gemini(msg) for msg in messages)

async def edit_or_reply_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, processing_msg_id: int = None, parse_mode=None, disable_web_page_preview=True):
    """Tries to edit a message, falls back to sending a new one if edit fails or no ID provided."""
    try:
        if processing_msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )
            return # Success editing
        # If no processing_msg_id, fall through to send new message
    except TelegramError as e:
        # Log the edit failure specifically
        logger.error(f"Failed to edit message {processing_msg_id} in chat {chat_id}: {e}. Attempting to send new message instead.")
        # Fall through to send new message

    # Send a new message if editing wasn't attempted or failed
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview
        )
    except TelegramError as send_err:
        # Log failure to send even the new message
        logger.error(f"Failed even to send new message to chat {chat_id} after edit failed or wasn't applicable: {send_err}")


# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start or /help command."""
    if not update.message:
        return # Should not happen for command handlers

    start_text = (
        "Hi! I'm a bot designed to summarize recent messages in this group.\n\n"
        f"Use the command `/{COMMAND_NAME} [N]` where `N` is the number of recent messages you want to summarize. "
        f"(Default: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}).\n"
        f"Example: `/{COMMAND_NAME} 50`\n\n"
        f"I use the `{GEMINI_MODEL_NAME}` model for generating summaries.\n"
        f"There's a {SUMMARY_COOLDOWN_SECONDS}-second cooldown per user for this command to prevent spam.\n\n"
        "**Important:** For me to see messages and summarize them, 'Group Privacy' mode must be **disabled** in my settings. "
        "You can manage this via @BotFather (`/mybots` -> select bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
    )
    await update.message.reply_text(
        start_text,
        parse_mode=constants.ParseMode.MARKDOWN,
        disable_web_page_preview=True
    )

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /summarize command to generate and send a summary."""
    if not update.message or not update.message.from_user or not update.message.chat:
        logger.warning("Summarize command received without essential message/user/chat info.")
        return # Cannot proceed without core info

    message = update.message
    user = message.from_user
    chat = message.chat
    chat_id = chat.id
    user_id = user.id

    # 1. Check command origin (only groups/supergroups)
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        logger.debug(f"Command '/{COMMAND_NAME}' ignored from non-group chat {chat_id} type {chat.type} by user {user_id}")
        # Optionally reply in private chat that it only works in groups
        # await context.bot.send_message(user_id, "Sorry, the /summarize command only works in group chats.")
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
                disable_notification=True, # Be less intrusive
            )
        except TelegramError as e:
             logger.warning(f"Failed to send rate limit message to user {user_id} in chat {chat_id}: {e}")
        return

    # 3. Parse arguments
    num_messages = DEFAULT_SUMMARY_MESSAGES # Default value
    if context.args:
        try:
            requested_num = int(context.args[0])
            if 0 < requested_num <= MAX_SUMMARY_MESSAGES:
                num_messages = requested_num
            else:
                await message.reply_text(
                    f"Please provide a number between 1 and {MAX_SUMMARY_MESSAGES}. "
                    f"Usage: `/{COMMAND_NAME} [number]` (Using default {DEFAULT_SUMMARY_MESSAGES})",
                     parse_mode=constants.ParseMode.MARKDOWN,
                )
                # Continue with default, but inform user
                num_messages = DEFAULT_SUMMARY_MESSAGES
        except (ValueError, IndexError):
             await message.reply_text(
                 f"Invalid number format. Using default {DEFAULT_SUMMARY_MESSAGES}. "
                 f"Usage: `/{COMMAND_NAME} [number]` (e.g., `/{COMMAND_NAME} 50`).",
                  parse_mode=constants.ParseMode.MARKDOWN,
             )
             # Continue with default
             num_messages = DEFAULT_SUMMARY_MESSAGES
    # If no args, num_messages remains DEFAULT_SUMMARY_MESSAGES

    # 4. Retrieve messages from cache
    chat_messages_deque = message_cache.get(chat_id)
    if not chat_messages_deque:
        logger.info(f"No message cache found for chat {chat_id} when user {user_id} requested summarization.")
        await message.reply_text(
            "I haven't stored any messages from this chat yet, or my cache was cleared. "
            "Please wait for more messages to arrive or ensure I have permission to read messages (Group Privacy off)."
        )
        return

    # Create a snapshot for processing to avoid race conditions if cache updates during processing
    with context.application._lock: # Use application lock for potential thread safety if needed elsewhere
        chat_messages = list(chat_messages_deque)

    if not chat_messages:
        logger.info(f"Message cache for chat {chat_id} is empty when user {user_id} requested summarization.")
        await message.reply_text("My message cache for this chat is currently empty. Please wait for new messages.")
        return

    # Select the most recent N messages
    messages_to_summarize = chat_messages[-num_messages:]
    actual_count = len(messages_to_summarize)

    if actual_count == 0: # Should be unlikely after previous checks, but good failsafe
        logger.warning(f"Attempted to summarize 0 messages for chat {chat_id} requested by {user_id}, despite non-empty cache.")
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
        # Store message ID for potential editing later
        processing_msg_id = processing_msg.message_id if processing_msg else None
    except TelegramError as send_err:
        logger.error(f"Failed to send 'processing' message to chat {chat_id} for user {user_id}: {send_err}")
        # If we can't send status, we probably can't send the result either. Abort early.
        return

    # 6. Format prompt and call Gemini API
    formatted_text = format_messages_for_prompt(messages_to_summarize)
    # Check length again, potentially adjusting based on newer model limits if known
    if len(formatted_text) > 32000: # Example limit adjustment if needed for 1.5 flash/pro
        logger.warning(f"Formatted text for chat {chat_id} exceeds 32,000 chars ({len(formatted_text)}). May be truncated by API or cause errors.")
        # Optionally truncate here or inform user more clearly

    # Define the prompt (could be moved to a constant or config)
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
    # Default error message, can be overridden by specific exceptions
    user_error_message = "❌ Oops! Something went wrong while generating the summary. Please try again later. If the problem persists, contact the bot admin."

    try:
        # Configure generation parameters per-request if needed (e.g., temperature)
        generation_config = GenerationConfig(
            # temperature=0.7,
            # max_output_tokens=1024, # Limit summary length if desired
        )

        # Run blocking API call in a separate thread to avoid blocking the event loop
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=generation_config,
            # Safety settings are usually set at model level, but can be overridden here
            # request_options={'timeout': 60} # Set API call timeout if needed
        )

        # --- Refined Response Handling (for v0.8.5+) ---
        if not response.candidates:
             block_reason = "Unknown reason (No candidates)"
             # Safely access prompt_feedback and block_reason
             prompt_feedback = getattr(response, 'prompt_feedback', None)
             if prompt_feedback and hasattr(prompt_feedback, 'block_reason'):
                 block_reason_enum = getattr(prompt_feedback, 'block_reason', None)
                 block_reason = getattr(block_reason_enum, 'name', str(block_reason_enum)) # Get name if enum, else string
             logger.warning(f"Summary generation blocked for chat {chat_id} by user {user_id}. Reason: {block_reason}. Prompt length: {len(prompt)}. Feedback: {prompt_feedback}")
             user_error_message = f"❌ Summary generation failed. The request was blocked, possibly due to safety filters ({block_reason}). Please revise the conversation if sensitive content is present."
             raise BlockedPromptException(f"Blocked prompt: {block_reason}")

        candidate = response.candidates[0] # Assume at least one candidate if not blocked

        # --- Standard Enum Check (for v0.8.5+) ---
        finish_reason = getattr(candidate, 'finish_reason', None) # Safely get finish_reason

        # Compare finish_reason against known enum values
        # Make sure genai.types.FinishReason is the correct path in v0.8.5
        if finish_reason == genai.types.FinishReason.SAFETY:
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in safety_ratings])
            citation_metadata = getattr(candidate, 'citation_metadata', None)
            logger.warning(f"Summary generation stopped due to SAFETY concerns for chat {chat_id} requested by {user_id}. Ratings: {safety_ratings_str}. Citations: {citation_metadata}")
            user_error_message = "❌ Summary generation stopped due to safety concerns. The generated content might contain sensitive topics based on safety ratings."
            raise StopCandidateException(f"Safety stop: {safety_ratings_str}")

        elif finish_reason == genai.types.FinishReason.RECITATION:
             citation_metadata = getattr(candidate, 'citation_metadata', None)
             logger.warning(f"Summary generation stopped due to RECITATION concerns for chat {chat_id} requested by {user_id}. Citations: {citation_metadata}")
             user_error_message = "❌ Summary generation stopped. The content may include material from protected sources."
             raise StopCandidateException("Recitation stop")

        elif finish_reason not in [genai.types.FinishReason.STOP, genai.types.FinishReason.MAX_TOKENS]:
             finish_reason_name = getattr(finish_reason, 'name', 'UNKNOWN') # Safely get enum name
             logger.warning(f"Summary generation finished unexpectedly for chat {chat_id} requested by {user_id}. Reason: {finish_reason_name}")
             user_error_message = f"❌ Summary generation finished unexpectedly ({finish_reason_name}). Please try again."
             raise StopCandidateException(f"Unexpected finish: {finish_reason_name}")
        # --- End of Standard Enum Check ---

        # Get the text content safely
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', []) if content else []

        if not parts:
             finish_reason_name = getattr(finish_reason, 'name', 'UNKNOWN') # Use already retrieved finish_reason
             logger.warning(f"Gemini API returned no content parts for chat {chat_id} requested by {user_id}. Finish Reason: {finish_reason_name}. Response: {response}")
             user_error_message = "❌ Summary generation resulted in no content. This might be due to filtering or an API issue. Please try again later."
             raise ValueError("Received no content/parts from API.")

        # Assuming text is in the first part if parts exist
        text_part = getattr(parts[0], 'text', None)
        if text_part is None: # Check specifically for None, allowing empty strings
             finish_reason_name = getattr(finish_reason, 'name', 'UNKNOWN')
             logger.warning(f"Gemini API returned empty text in content part for chat {chat_id} requested by {user_id}. Finish Reason: {finish_reason_name}. Response: {response}")
             user_error_message = "❌ Summary generation resulted in empty content. This might be due to filtering or an API issue. Please try again later."
             raise ValueError("Received empty text part from API.")

        summary_text = text_part.strip() # Strip whitespace from the result

        # Final check if the summary is empty after stripping
        if not summary_text:
             logger.warning(f"Gemini API returned whitespace-only summary for chat {chat_id} requested by {user_id}.")
             user_error_message = "❌ Summary generation resulted in an empty summary after processing."
             error_occurred = True # Treat as error for sending message logic

        # If successful *and* summary text is not empty, update rate limit timestamp
        if not error_occurred and summary_text:
            user_last_summary_request[user_id] = time.monotonic()

    # --- Exception Handling ---
    except (BlockedPromptException, StopCandidateException) as safety_exception:
        # Specific exceptions caught above, user message already set
        logger.error(f"{type(safety_exception).__name__} during summary for chat {chat_id}, user {user_id}: {safety_exception}", exc_info=False) # No need for full traceback here
        error_occurred = True

    except google.api_core.exceptions.InternalServerError as gemini_ise:
        logger.error(f"Gemini Internal Server Error during summary for chat {chat_id}, user {user_id}: {gemini_ise}", exc_info=True)
        user_error_message = "❌ The summarization service (Gemini) reported an internal error. Please try again later."
        error_occurred = True

    except google.api_core.exceptions.GoogleAPIError as api_error:
        # Catch other Google API errors (authentication, timeouts, quotas, etc.)
        logger.error(f"Google API Error during summary for chat {chat_id}, user {user_id}: {api_error}", exc_info=True)
        if isinstance(api_error, google.api_core.exceptions.PermissionDenied):
             user_error_message = "❌ Permission denied by the summarization service. Please check the API key and project settings."
        elif isinstance(api_error, google.api_core.exceptions.DeadlineExceeded):
             user_error_message = "❌ The request to the summarization service timed out. Please try again later."
        elif isinstance(api_error, google.api_core.exceptions.Unauthenticated):
             user_error_message = "❌ Authentication error with the summarization service. Please notify the bot admin."
        elif isinstance(api_error, google.api_core.exceptions.ResourceExhausted): # 429 Too Many Requests
             user_error_message = "❌ Rate limit exceeded for the summarization service. Please wait and try again."
        else: # Catch-all for other Google API errors
             user_error_message = f"❌ A Google API error occurred ({type(api_error).__name__}). Please notify the bot admin."
        error_occurred = True

    except ValueError as val_err: # Catch specific ValueErrors raised during response processing
        logger.error(f"ValueError during summary processing for chat {chat_id}, user {user_id}: {val_err}", exc_info=False)
        # user_error_message is already set when the error was raised
        error_occurred = True

    except Exception as e:
        # Catch any other unexpected Python errors during summary generation
        logger.error(f"Unexpected Python error during summary generation/processing for chat {chat_id}, user {user_id}: {e}", exc_info=True)
        # Keep the default generic user_error_message
        error_occurred = True
        # Notify admin if configured
        if ADMIN_CHAT_ID:
            try:
                # Send a concise error report to admin
                error_details = (
                    f"Unexpected error in summarize_command for chat {chat_id}, user {user_id}:\n"
                    f"Type: {type(e).__name__}\nError: {e}\n"
                    f"Traceback:\n{traceback.format_exc(limit=5)}" # Limit traceback depth
                )
                await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=error_details[:4000]) # Limit message length
            except Exception as notify_err:
                logger.error(f"Failed to send error notification to admin {ADMIN_CHAT_ID}: {notify_err}")

    # 7. Send Result or Error Message
    # This block executes regardless of whether an error occurred above
    if not error_occurred and summary_text:
        # Success path: Construct and send the summary
        try:
            first_msg_id = messages_to_summarize[0][0]
            last_msg_id = messages_to_summarize[-1][0]
            reply_text = (
                f"✨ **Summary of last {actual_count} messages (approx. ID {first_msg_id} to {last_msg_id}):** ✨\n\n"
                f"{summary_text}\n\n"
                f"*Summary generated by `{GEMINI_MODEL_NAME}`. Message IDs are for reference.*"
            )
            await edit_or_reply_message(context, chat_id, reply_text, processing_msg_id, constants.ParseMode.MARKDOWN)
            logger.info(f"Successfully generated and sent summary for chat {chat_id} ({actual_count} messages) requested by user {user_id}.")
        except Exception as final_send_err:
            # Log error even during final message sending/editing attempt
            logger.error(f"Error sending final summary message for chat {chat_id}: {final_send_err}", exc_info=True)
            # Optionally try sending a simple error message if the summary send failed
            await edit_or_reply_message(context, chat_id, "❌ Error displaying the summary.", processing_msg_id)

    elif error_occurred:
        # Error path: Send the specific user_error_message determined during exception handling
        await edit_or_reply_message(context, chat_id, user_error_message, processing_msg_id)
        logger.warning(f"Summary failed for chat {chat_id} requested by user {user_id}. Sent error/info message to user.")
    # Implicit else: If no error occurred but summary_text ended up empty (whitespace only), the warning is logged,
    # and the user_error_message for that case is sent.


# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stores incoming text messages from groups/supergroups in the cache."""
    if not update.message or not update.message.text or update.message.via_bot:
        # Ignore non-text messages, messages without text, or messages from bots
        return

    chat = update.message.chat
    # Only cache messages from group chats
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        return

    chat_id = chat.id
    user = update.message.from_user
    message = update.message

    # Determine user name safely
    if not user:
        user_name = "Anonymous" # Handle cases like anonymous admins
    else:
        user_name = user.first_name.strip() if user.first_name else ""
        if not user_name and user.username:
            user_name = user.username.strip()
        if not user_name: # Fallback if no first name or username
            user_name = f"User_{user.id}"

    message_id = message.message_id
    text = message.text
    # Ensure timestamp is timezone-aware UTC
    timestamp = message.date if message.date else datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
         timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Add message data tuple to this chat's cache deque
    try:
        message_data = (message_id, user_name, text, timestamp)
        # defaultdict automatically creates deque if it doesn't exist
        message_cache[chat_id].append(message_data)
        # Optional: Debug log for frequent cache monitoring
        # logger.debug(f"Cached msg {message_id} from '{user_name}' in chat {chat_id}. Cache size: {len(message_cache[chat_id])}")
    except Exception as cache_err:
        logger.error(f"Failed to cache message {message_id} for chat {chat_id}: {cache_err}", exc_info=True)


# --- Error Handler ---

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and notify admin if configured."""
    # Log the error with traceback
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    # Notify admin if configured
    if ADMIN_CHAT_ID:
        try:
            # Format a concise error message for admin
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__, limit=5) # Limit traceback depth
            tb_string = "".join(tb_list)
            # Try to get update details safely
            update_str = str(update) if update else "N/A"
            error_message = (
                f"⚠️ BOT ERROR ⚠️\n\n"
                f"Error Type: {type(context.error).__name__}\n"
                f"Error: {context.error}\n\n"
                f"Update: {update_str[:500]}...\n\n" # Limit update string length
                f"Traceback (limited):\n```\n{tb_string[:3000]}\n```" # Limit traceback length
            )
            # Send error to admin chat, ensuring it fits within Telegram limits
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=error_message[:4096], # Telegram message length limit
                parse_mode=constants.ParseMode.MARKDOWN
            )
        except Exception as notify_err:
            # Log failure to notify admin
            logger.error(f"CRITICAL: Failed to send error notification to admin {ADMIN_CHAT_ID}: {notify_err}")


# --- Main Application Setup ---

def main() -> None:
    """Sets up the Telegram Application and starts the bot polling."""
    logger.info(f"--- Initializing Telegram Bot (PID: {os.getpid()}) ---")
    logger.info(f"Using Telegram Bot Token: {'*' * (len(TELEGRAM_TOKEN or '') - 4)}{(TELEGRAM_TOKEN or '')[-4:]}")
    logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    logger.info(f"Message cache size per chat: {MESSAGE_CACHE_SIZE}")
    logger.info(f"Default summary length: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}")
    logger.info(f"Summary command cooldown: {SUMMARY_COOLDOWN_SECONDS} seconds per user")
    if ADMIN_CHAT_ID:
        logger.info(f"Admin chat ID for notifications: {ADMIN_CHAT_ID}")
    else:
        logger.warning("ADMIN_CHAT_ID not set. Error notifications will not be sent via Telegram.")

    # Create the Telegram Application using ApplicationBuilder
    # Consider adding persistence later if needed (e.g., PicklePersistence)
    builder = ApplicationBuilder().token(TELEGRAM_TOKEN)
    # Optional: Configure connection pool size for potentially high load
    # builder.connection_pool_size(512)
    # Optional: Increase connect/read timeouts if experiencing frequent timeout errors
    # builder.connect_timeout(10.0).read_timeout(30.0)
    application = builder.build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", start_command)) # Alias help to start
    application.add_handler(CommandHandler(COMMAND_NAME, summarize_command))

    # Register message handler for caching relevant messages
    # Filters: TEXT only, not a command, only in groups/supergroups, not via another bot
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS & ~filters.VIA_BOT,
        handle_message
    ))

    # Register the global error handler
    application.add_error_handler(error_handler)

    # Start polling for updates
    logger.info("Bot polling started...")
    # Specify allowed updates to potentially improve efficiency if needed
    # application.run_polling(allowed_updates=[Update.MESSAGE], drop_pending_updates=True)
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True) # Process all types

    logger.info("--- Bot polling stopped ---")

if __name__ == "__main__":
    main()