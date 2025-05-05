# -*- coding: utf-8 -*-
import logging
import os
import asyncio
import time
import traceback
import html # For escaping HTML in error handler
import json # For potentially dumping update object
import google.api_core.exceptions
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

# Third-party libraries
import google.generativeai as genai
# Import necessary types for v0.8.5+
from google.generativeai.types import (
    GenerationConfig, SafetySettingDict, HarmCategory, HarmBlockThreshold,
    BlockedPromptException, StopCandidateException
)
# Language detection
from langdetect import detect, LangDetectException
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
# API timeout in seconds (default: 30 seconds)
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "30"))


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
    )
    logging.info(f"Successfully configured Gemini AI with model: {GEMINI_MODEL_NAME}")
except Exception as e:
    logging.critical(f"CRITICAL: Failed to configure or test Gemini AI: {e}", exc_info=True)
    import sys
    print(f"CRITICAL: Failed to configure Gemini: {e}", file=sys.stderr)
    sys.exit(1)

# Bot Configuration
COMMAND_NAME = "summarize"
DEFAULT_SUMMARY_MESSAGES = 25
MAX_SUMMARY_MESSAGES = 200
MESSAGE_CACHE_SIZE = 500
SUMMARY_COOLDOWN_SECONDS = 60

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%dT%H:%M:%S%z'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- In-Memory Caches ---
message_cache: dict[int, deque] = defaultdict(lambda: deque(maxlen=MESSAGE_CACHE_SIZE))
user_last_summary_request: dict[int, float] = {}
chat_language_cache: dict[int, str] = {}  # Store detected language by chat_id
last_cache_cleanup: float = time.monotonic()  # Time tracker for periodic cache cleanup
CACHE_CLEANUP_INTERVAL = 3600  # Clean up old entries once per hour (in seconds)

# --- Helper Functions ---

def sanitize_for_prompt(text: str) -> str:
    """Basic sanitization for user names included in prompts."""
    return text.replace('[', '(').replace(']', ')')

def format_message_for_gemini(msg_data: tuple) -> str:
    """Formats a single message tuple for the Gemini prompt."""
    message_id, user_name, text, timestamp = msg_data
    ts_str = timestamp.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    safe_user_name = sanitize_for_prompt(user_name)
    text_content = text if text is not None else ""
    return f"[{ts_str} - {safe_user_name} (ID: {message_id})]: {text_content}"

def format_messages_for_prompt(messages: list[tuple]) -> str:
    """Formats a list of message tuples into a single string for the Gemini prompt."""
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
            return
    except TelegramError as e:
        logger.error(f"Failed to edit message {processing_msg_id} in chat {chat_id}: {e}. Attempting to send new message instead.")

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            disable_web_page_preview=disable_web_page_preview
        )
    except TelegramError as send_err:
        logger.error(f"Failed even to send new message to chat {chat_id} after edit failed or wasn't applicable: {send_err}")

def detect_language(messages: list[tuple]) -> str:
    """Detect the most common language in a list of messages, supporting only English and Persian.
    Uses a weighted approach with confidence threshold for more reliable detection."""
    if not messages:
        return "en"  # Default to English if no messages
    
    # Collect texts with adequate length (longer texts are more reliable for detection)
    filtered_texts = [msg[2] for msg in messages if msg[2] and len(msg[2]) > 10]
    
    # Return English if not enough text to detect
    if not filtered_texts or sum(len(text) for text in filtered_texts) < 50:
        return "en"
        
    # Concatenate all messages into a single string for more accurate detection
    all_text = " ".join(filtered_texts)
    
    # Track detection attempts
    detection_attempts = []
    
    try:
        # Use langdetect with profile analysis for better accuracy
        from langdetect import detect_langs
        
        # Try to detect with full text first
        lang_results = detect_langs(all_text)
        
        # Check if Persian is detected with good confidence
        for lang in lang_results:
            if lang.lang in ["fa", "per", "pes", "ira"] and lang.prob > 0.5:
                logger.info(f"Detected Persian language with {lang.prob:.2f} confidence")
                return "fa"
            detection_attempts.append((lang.lang, lang.prob))
                
        # If no clear Persian detection but the top result is Persian with any confidence
        top_lang = lang_results[0] if lang_results else None
        if top_lang and top_lang.lang in ["fa", "per", "pes", "ira"]:
            logger.info(f"Detected Persian as top language with {top_lang.prob:.2f} confidence")
            return "fa"
            
        # Fall back to standard detection if needed
        detected = detect(all_text)
        if detected in ["fa", "per", "pes", "ira"]:
            logger.info(f"Detected Persian language using fallback method")
            return "fa"
        else:
            logger.info(f"Detected non-Persian language: {detected}, using English. Attempts: {detection_attempts}")
            return "en"
    except LangDetectException as e:
        logger.warning(f"Language detection error: {e}")
        return "en"  # Default to English on detection error
    except Exception as e:
        logger.warning(f"Unexpected error in language detection: {e}")
        return "en"  # Default to English on any error

def cleanup_old_cache_entries() -> None:
    """Periodically clean up old entries from caches to prevent memory growth."""
    global last_cache_cleanup
    
    # Only run cleanup if enough time has passed since last cleanup
    now = time.monotonic()
    if now - last_cache_cleanup < CACHE_CLEANUP_INTERVAL:
        return
        
    try:
        # Clean up user rate limiting cache - remove entries older than 2 hours
        cleanup_threshold = now - 7200  # 2 hours in seconds
        users_to_remove = [
            user_id for user_id, timestamp in user_last_summary_request.items() 
            if timestamp < cleanup_threshold
        ]
        for user_id in users_to_remove:
            user_last_summary_request.pop(user_id, None)
            
        # Clean up empty caches or those for inactive chats (no messages in 24 hours)
        inactive_threshold = time.time() - 86400  # 24 hours in seconds
        chats_to_remove = []
        for chat_id, msg_deque in message_cache.items():
            if not msg_deque:  # Empty cache
                chats_to_remove.append(chat_id)
                continue
                
            # Get timestamp of last message (if available)
            try:
                last_msg = msg_deque[-1]
                if len(last_msg) >= 4:  # Ensure tuple has timestamp field
                    last_timestamp = last_msg[3]  # Access the timestamp
                    if isinstance(last_timestamp, datetime):
                        # Convert to epoch seconds for comparison
                        last_activity = last_timestamp.timestamp()
                        if last_activity < inactive_threshold:
                            chats_to_remove.append(chat_id)
            except (IndexError, TypeError, AttributeError) as e:
                logger.debug(f"Error checking message cache timestamps for chat {chat_id}: {e}")
        
        # Remove inactive chats from all caches
        for chat_id in chats_to_remove:
            message_cache.pop(chat_id, None)
            chat_language_cache.pop(chat_id, None)
            
        if users_to_remove or chats_to_remove:
            logger.info(f"Cache cleanup: removed {len(users_to_remove)} user entries and {len(chats_to_remove)} chat entries")
            
        # Update last cleanup time
        last_cache_cleanup = now
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}", exc_info=True)

# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start or /help command."""
    if not update.message:
        return

    # Get user locale from Telegram
    user_lang_code = update.effective_user.language_code if update.effective_user else None
    
    # Only support English and Persian
    lang = "fa" if user_lang_code == "fa" else "en"
    
    # Multi-language start messages
    start_texts = {
        "en": (
            "Hi! I'm a bot designed to summarize recent messages in this group.\n\n"
            f"Use the command `/{COMMAND_NAME} [N]` where `N` is the number of recent messages you want to summarize. "
            f"(Default: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}).\n"
            f"Example: `/{COMMAND_NAME} 50`\n\n"
            f"I use the `{GEMINI_MODEL_NAME}` model for generating summaries.\n"
            f"There's a {SUMMARY_COOLDOWN_SECONDS}-second cooldown per user for this command to prevent spam.\n\n"
            "**Important:** For me to see messages and summarize them, 'Group Privacy' mode must be **disabled** in my settings. "
            "You can manage this via @BotFather (`/mybots` -> select bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
        ),
        "fa": (
            "سلام! من یک ربات هستم که برای خلاصه‌سازی پیام‌های اخیر در این گروه طراحی شده‌ام.\n\n"
            f"از دستور `/{COMMAND_NAME} [N]` استفاده کنید که در آن `N` تعداد پیام‌های اخیری است که می‌خواهید خلاصه شود. "
            f"(پیش‌فرض: {DEFAULT_SUMMARY_MESSAGES}، حداکثر: {MAX_SUMMARY_MESSAGES}).\n"
            f"مثال: `/{COMMAND_NAME} 50`\n\n"
            f"من از مدل `{GEMINI_MODEL_NAME}` برای تولید خلاصه‌ها استفاده می‌کنم.\n"
            f"یک زمان انتظار {SUMMARY_COOLDOWN_SECONDS} ثانیه‌ای برای هر کاربر برای این دستور وجود دارد تا از اسپم جلوگیری شود.\n\n"
            "**مهم:** برای اینکه من بتوانم پیام‌ها را ببینم و خلاصه کنم، حالت 'حریم خصوصی گروه' باید در تنظیمات من **غیرفعال** باشد. "
            "شما می‌توانید این را از طریق @BotFather مدیریت کنید (`/mybots` -> انتخاب ربات -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
        )
    }
    
    await update.message.reply_text(
        start_texts.get(lang, start_texts["en"]),
        parse_mode=constants.ParseMode.MARKDOWN,
        disable_web_page_preview=True
    )

async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /summarize command to generate and send a summary."""
    if not update.message or not update.message.from_user or not update.message.chat:
        logger.warning("Summarize command received without essential message/user/chat info.")
        return

    message = update.message
    user = message.from_user
    chat = message.chat
    chat_id = chat.id
    user_id = user.id

    # 1. Check command origin
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        logger.debug(f"Command '/{COMMAND_NAME}' ignored from non-group chat {chat_id} type {chat.type} by user {user_id}")
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

    # 3. Parse arguments with improved validation
    num_messages = DEFAULT_SUMMARY_MESSAGES
    if context.args:
        try:
            arg = context.args[0].strip()
            
            # Check for non-numeric inputs
            if not arg.isdigit():
                raise ValueError(f"Non-numeric argument: {arg}")
                
            requested_num = int(arg)
            
            # Validate range
            if requested_num <= 0:
                await message.reply_text(
                    f"Please provide a number greater than 0. "
                    f"Using default {DEFAULT_SUMMARY_MESSAGES} messages.",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                num_messages = DEFAULT_SUMMARY_MESSAGES
            elif requested_num > MAX_SUMMARY_MESSAGES:
                await message.reply_text(
                    f"The maximum allowed is {MAX_SUMMARY_MESSAGES} messages. "
                    f"Using {MAX_SUMMARY_MESSAGES} messages.",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                num_messages = MAX_SUMMARY_MESSAGES
            else:
                num_messages = requested_num
                
            logger.debug(f"User {user_id} requested summary of {num_messages} messages in chat {chat_id}")
                
        except ValueError as e:
            logger.debug(f"Invalid number format from user {user_id} in chat {chat_id}: {e}")
            await message.reply_text(
                f"Invalid number format. Using default {DEFAULT_SUMMARY_MESSAGES}. "
                f"Usage: `/{COMMAND_NAME} [number]` (e.g., `/{COMMAND_NAME} 50`).",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            num_messages = DEFAULT_SUMMARY_MESSAGES
        except IndexError:
            # This shouldn't normally happen if context.args exists
            logger.warning(f"Unexpected IndexError when parsing args from user {user_id} in chat {chat_id}")
            num_messages = DEFAULT_SUMMARY_MESSAGES

    # 4. Retrieve messages from cache
    chat_messages_deque = message_cache.get(chat_id)
    if not chat_messages_deque:
        logger.info(f"No message cache found for chat {chat_id} when user {user_id} requested summarization.")
        await message.reply_text(
            "I haven't stored any messages from this chat yet, or my cache was cleared. "
            "Please wait for more messages to arrive or ensure I have permission to read messages (Group Privacy off)."
        )
        return

    # Create a snapshot for processing. The lock was removed as it's not standard/needed.
    chat_messages = list(chat_messages_deque)

    if not chat_messages:
        logger.info(f"Message cache for chat {chat_id} is empty when user {user_id} requested summarization.")
        await message.reply_text("My message cache for this chat is currently empty. Please wait for new messages.")
        return

    messages_to_summarize = chat_messages[-num_messages:]
    actual_count = len(messages_to_summarize)

    if actual_count == 0:
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
        processing_msg_id = processing_msg.message_id if processing_msg else None
    except TelegramError as send_err:
        logger.error(f"Failed to send 'processing' message to chat {chat_id} for user {user_id}: {send_err}")
        return

    # 6. Format prompt and call Gemini API
    formatted_text = format_messages_for_prompt(messages_to_summarize)
    if len(formatted_text) > 32000:
        logger.warning(f"Formatted text for chat {chat_id} exceeds 32,000 chars ({len(formatted_text)}). May be truncated by API or cause errors.")

    # Detect language if not already cached
    chat_lang = chat_language_cache.get(chat_id, None)
    if not chat_lang:
        chat_lang = detect_language(messages_to_summarize)
        chat_language_cache[chat_id] = chat_lang
        logger.info(f"Detected language for chat {chat_id}: {chat_lang}")
    
    # Multi-language prompt templates
    prompt_templates = {
        "en": {
            "intro": "You are a helpful assistant tasked with summarizing Telegram group chat conversations.\n"
                    "Provide a *concise, well-structured, and clearly formatted* summary of the following messages. "
                    "Focus on key discussion points, decisions made, questions asked, and any action items mentioned.\n\n"
                    "Structure your summary as follows:\n"
                    "1. Start with a very brief overall summary in 1-2 sentences\n"
                    "2. Organize content into clearly labeled topics or themes using bold headings with emoji prefixes\n"
                    "3. Under each topic, use bullet points to list key points\n"
                    "4. Format important information, names, and numbers in *bold* or _italic_ for emphasis\n\n"
                    "DO NOT reference message IDs or include citations. Focus only on summarizing the content in a readable, well-structured format.",
            "summary_request": "Concise Summary:",
            "processing_message": f"⏳ Fetching and summarizing the last {actual_count} cached messages using AI... please wait.",
            "error_message": "❌ Oops! Something went wrong while generating the summary. Please try again later. If the problem persists, contact the bot admin.",
            "summary_header": f"**✨ SUMMARY OF RECENT MESSAGES ✨**\n\n",
            "summary_footer": ""
        },
        "fa": {
            "intro": "شما یک دستیار هوشمند هستید که وظیفه خلاصه‌سازی گفتگوهای گروهی تلگرام را دارید.\n"
                    "یک خلاصه *موجز، ساختارمند و به خوبی قالب‌بندی شده* از پیام‌های زیر ارائه دهید. "
                    "بر نکات کلیدی بحث، تصمیمات گرفته شده، سؤالات مطرح شده، و هرگونه موارد عملی ذکر شده تمرکز کنید.\n\n"
                    "خلاصه خود را به این شکل ساختاربندی کنید:\n"
                    "1. با یک خلاصه کلی و بسیار مختصر در 1-2 جمله شروع کنید\n"
                    "2. محتوا را به موضوعات یا تم‌های مشخص با عناوین پررنگ و ایموجی مناسب سازماندهی کنید\n"
                    "3. زیر هر موضوع، از نقاط گلوله‌ای برای لیست کردن نکات کلیدی استفاده کنید\n"
                    "4. اطلاعات مهم، نام‌ها و اعداد را با فرمت *پررنگ* یا _مورب_ برای تأکید قالب‌بندی کنید\n\n"
                    "به شناسه پیام‌ها اشاره نکنید و مرجع‌دهی نداشته باشید. فقط بر خلاصه‌سازی محتوا در قالبی خوانا و ساختارمند تمرکز کنید.",
            "summary_request": "خلاصه موجز:",
            "processing_message": f"⏳ در حال دریافت و خلاصه‌سازی {actual_count} پیام اخیر با استفاده از هوش مصنوعی... لطفاً صبر کنید.",
            "error_message": "❌ اوپس! هنگام تولید خلاصه مشکلی پیش آمد. لطفاً بعداً دوباره امتحان کنید. اگر مشکل ادامه داشت، با مدیر ربات تماس بگیرید.",
            "summary_header": f"**✨ خلاصه پیام‌های اخیر ✨**\n\n",
            "summary_footer": ""
        }
    }
    
    # Use English templates as fallback for unsupported languages
    templates = prompt_templates.get(chat_lang, prompt_templates["en"])
    
    # Update processing message with language-specific version
    try:
        if processing_msg_id:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=processing_msg_id,
                text=templates["processing_message"],
                parse_mode=constants.ParseMode.MARKDOWN
            )
    except TelegramError as edit_err:
        logger.warning(f"Failed to update processing message with language-specific version: {edit_err}")
    
    # Build the language-specific prompt
    prompt = (
        f"{templates['intro']}\n\n"
        "--- Conversation Start ---\n"
        f"{formatted_text}\n"
        "--- Conversation End ---\n\n"
        f"{templates['summary_request']}"
    )

    summary_text = ""
    error_occurred = False
    user_error_message = templates["error_message"]

    try:
        generation_config = GenerationConfig() # Add specific config if needed

        # Add timeout to prevent hanging on slow responses
        response = await asyncio.wait_for(
            asyncio.to_thread(
                gemini_model.generate_content,
                prompt,
                generation_config=generation_config,
            ),
            timeout=API_TIMEOUT_SECONDS
        )

        # --- Refined Response Handling (for v0.8.5+) ---
        if not response.candidates:
             block_reason = "Unknown reason (No candidates)"
             prompt_feedback = getattr(response, 'prompt_feedback', None)
             if prompt_feedback and hasattr(prompt_feedback, 'block_reason'):
                 block_reason_enum = getattr(prompt_feedback, 'block_reason', None)
                 block_reason = getattr(block_reason_enum, 'name', str(block_reason_enum))
             logger.warning(f"Summary generation blocked for chat {chat_id} by user {user_id}. Reason: {block_reason}. Prompt length: {len(prompt)}. Feedback: {prompt_feedback}")
             user_error_message = f"❌ Summary generation failed. The request was blocked, possibly due to safety filters ({block_reason}). Please revise the conversation if sensitive content is present."
             raise BlockedPromptException(f"Blocked prompt: {block_reason}")

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)

        # --- Standard Enum Check (for v0.8.5+) ---
        if finish_reason == "SAFETY":
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            # Handle both string and enum safety ratings formats
            safety_ratings_str = ""
            try:
                # Try to handle different possible formats of safety ratings
                if safety_ratings:
                    ratings_parts = []
                    for rating in safety_ratings:
                        category = getattr(rating, 'category', None)
                        probability = getattr(rating, 'probability', None)
                        
                        # Get names from objects or use values directly if they're strings
                        category_name = getattr(category, 'name', str(category)) if category else 'Unknown'
                        probability_name = getattr(probability, 'name', str(probability)) if probability else 'Unknown'
                        
                        ratings_parts.append(f"{category_name}: {probability_name}")
                    
                    safety_ratings_str = ", ".join(ratings_parts)
                else:
                    safety_ratings_str = "No detailed ratings available"
            except Exception as e:
                safety_ratings_str = f"Error parsing ratings: {e}"
                
            citation_metadata = getattr(candidate, 'citation_metadata', None)
            logger.warning(f"Summary generation stopped due to SAFETY concerns for chat {chat_id} requested by {user_id}. Ratings: {safety_ratings_str}. Citations: {citation_metadata}")
            user_error_message = "❌ Summary generation stopped due to safety concerns. The generated content might contain sensitive topics based on safety ratings."
            raise StopCandidateException(f"Safety stop: {safety_ratings_str}")

        elif finish_reason == "RECITATION":
             citation_metadata = getattr(candidate, 'citation_metadata', None)
             logger.warning(f"Summary generation stopped due to RECITATION concerns for chat {chat_id} requested by {user_id}. Citations: {citation_metadata}")
             user_error_message = "❌ Summary generation stopped. The content may include material from protected sources."
             raise StopCandidateException("Recitation stop")

        elif finish_reason not in ["STOP", "MAX_TOKENS"]:
             finish_reason_name = finish_reason if isinstance(finish_reason, str) else 'UNKNOWN'
             logger.warning(f"Summary generation received unexpected finish reason for chat {chat_id} requested by {user_id}. Reason: {finish_reason_name}")
             
             # Only treat truly problematic reasons as errors
             if finish_reason in ["SAFETY", "RECITATION", "BLOCKED"]:
                 user_error_message = f"❌ Summary generation stopped ({finish_reason_name}). Please try again."
                 raise StopCandidateException(f"Problematic finish: {finish_reason_name}")
             
             # For unknown or null finish reasons, continue processing
             # This handles cases where the API might return None or new finish reasons
             logger.info(f"Proceeding with content despite unexpected finish reason: {finish_reason_name}")
             # Not raising an exception here, continue with content processing

        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', []) if content else []

        if not parts:
             finish_reason_name = finish_reason if isinstance(finish_reason, str) else 'UNKNOWN'
             logger.warning(f"Gemini API returned no content parts for chat {chat_id} requested by {user_id}. Finish Reason: {finish_reason_name}. Response: {response}")
             user_error_message = "❌ Summary generation resulted in no content. This might be due to filtering or an API issue. Please try again later."
             raise ValueError("Received no content/parts from API.")

        text_part = getattr(parts[0], 'text', None)
        if text_part is None:
             finish_reason_name = finish_reason if isinstance(finish_reason, str) else 'UNKNOWN'
             logger.warning(f"Gemini API returned empty text in content part for chat {chat_id} requested by {user_id}. Finish Reason: {finish_reason_name}. Response: {response}")
             user_error_message = "❌ Summary generation resulted in empty content. This might be due to filtering or an API issue. Please try again later."
             raise ValueError("Received empty text part from API.")

        summary_text = text_part.strip()

        if not summary_text:
             logger.warning(f"Gemini API returned whitespace-only summary for chat {chat_id} requested by {user_id}.")
             user_error_message = "❌ Summary generation resulted in an empty summary after processing."
             error_occurred = True

        if not error_occurred and summary_text:
            user_last_summary_request[user_id] = time.monotonic()

    # --- Exception Handling ---
    except asyncio.TimeoutError:
        logger.error(f"Timeout ({API_TIMEOUT_SECONDS}s) exceeded waiting for Gemini API response for chat {chat_id}, user {user_id}")
        user_error_message = f"❌ The request to the summarization service timed out after {API_TIMEOUT_SECONDS} seconds. Please try again later."
        error_occurred = True
        
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
             user_error_message = f"❌ A Google API error occurred ({type(api_error).__name__}). Please notify the bot admin."
        error_occurred = True

    except ValueError as val_err:
        logger.error(f"ValueError during summary processing for chat {chat_id}, user {user_id}: {val_err}", exc_info=False)
        error_occurred = True

    except Exception as e:
        logger.error(f"Unexpected Python error during summary generation/processing for chat {chat_id}, user {user_id}: {e}", exc_info=True)
        error_occurred = True
        if ADMIN_CHAT_ID:
            try:
                error_details = (
                    f"Unexpected error in summarize_command for chat {chat_id}, user {user_id}:\n"
                    f"Type: {type(e).__name__}\nError: {e}\n"
                    f"Traceback:\n{traceback.format_exc(limit=5)}"
                )
                # Send plain text to admin to avoid parsing errors
                await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=error_details[:4000])
            except Exception as notify_err:
                logger.error(f"Failed to send error notification to admin {ADMIN_CHAT_ID}: {notify_err}")

    # 7. Send Result or Error Message
    if not error_occurred and summary_text:
        try:
            reply_text = (
                f"{templates['summary_header']}"
                f"{summary_text}\n\n"
                f"{templates['summary_footer']}"
            )
            await edit_or_reply_message(context, chat_id, reply_text, processing_msg_id, constants.ParseMode.MARKDOWN)
            logger.info(f"Successfully generated and sent summary for chat {chat_id} ({actual_count} messages) requested by user {user_id}.")
        except Exception as final_send_err:
            logger.error(f"Error sending final summary message for chat {chat_id}: {final_send_err}", exc_info=True)
            await edit_or_reply_message(context, chat_id, "❌ Error displaying the summary.", processing_msg_id)

    elif error_occurred:
        await edit_or_reply_message(context, chat_id, user_error_message, processing_msg_id)
        logger.warning(f"Summary failed for chat {chat_id} requested by user {user_id}. Sent error/info message to user.")

# --- Message Handler ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stores incoming text messages from groups/supergroups in the cache."""
    if not update.message or not update.message.text or update.message.via_bot:
        return

    # Periodically clean up old cache entries
    cleanup_old_cache_entries()

    chat = update.message.chat
    if chat.type not in (constants.ChatType.GROUP, constants.ChatType.SUPERGROUP):
        return

    chat_id = chat.id
    user = update.message.from_user
    message = update.message

    if not user:
        user_name = "Anonymous"
    else:
        user_name = user.first_name.strip() if user.first_name else ""
        if not user_name and user.username:
            user_name = user.username.strip()
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
        
        # Update language detection every 10 messages
        if len(message_cache[chat_id]) % 10 == 0 and len(message_cache[chat_id]) >= 10:
            # Get the last 20 messages or however many are available
            recent_messages = list(message_cache[chat_id])[-20:]
            lang = detect_language(recent_messages)
            if lang:
                chat_language_cache[chat_id] = lang
                logger.debug(f"Updated language for chat {chat_id} to {lang}")
    except Exception as cache_err:
        logger.error(f"Failed to cache message {message_id} for chat {chat_id}: {cache_err}", exc_info=True)

# --- Error Handler ---

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and notify admin if configured."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    if ADMIN_CHAT_ID:
        try:
            # Format a concise error message for admin
            tb_list = traceback.format_exception(None, context.error, context.error.__traceback__, limit=5)
            tb_string = "".join(tb_list)

            # Try to get update details safely, converting to dict if possible
            update_str = "N/A"
            if isinstance(update, Update):
                try:
                    # Convert Update to dict, then to JSON string for safer display
                    update_dict = update.to_dict()
                    update_str = json.dumps(update_dict, indent=2, ensure_ascii=False, default=str) # Use default=str for non-serializable objects
                except Exception as json_err:
                    logger.warning(f"Could not serialize update object to JSON: {json_err}")
                    update_str = str(update) # Fallback to string representation
            elif update:
                update_str = str(update)

            error_message = (
                f"⚠️ BOT ERROR ⚠️\n\n"
                f"Error Type: {type(context.error).__name__}\n"
                f"Error: {html.escape(str(context.error))}\n\n" # Escape potential HTML in error message
                f"Update (limited):\n<pre>{html.escape(update_str[:500])}...</pre>\n\n" # Use <pre> and escape update string
                f"Traceback (limited):\n<pre>{html.escape(tb_string[:3000])}</pre>" # Use <pre> and escape traceback
            )

            # Send error to admin chat using HTML parse mode for <pre> tags
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=error_message[:4096], # Telegram message length limit
                parse_mode=constants.ParseMode.HTML # Use HTML for <pre> tags
            )
        except Exception as notify_err:
            logger.error(f"CRITICAL: Failed to send error notification to admin {ADMIN_CHAT_ID}: {notify_err}")
            # Fallback: Try sending a very basic plain text notification if the formatted one failed
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=f"BOT ERROR: {type(context.error).__name__} - {context.error}. Check logs for details."
                )
            except Exception as fallback_err:
                 logger.error(f"CRITICAL: Failed even to send fallback error notification to admin: {fallback_err}")


# --- Main Application Setup ---

def main() -> None:
    """Sets up the Telegram Application and starts the bot polling."""
    # Display bot startup banner
    startup_banner = f"""
    ┌───────────────────────────────────────────────┐
    │ 🤖 Telegram Group Chat Summarizer Bot 🤖       │
    │ Using Google Gemini AI for summarization      │
    │ Version: 1.1.0                                │
    └───────────────────────────────────────────────┘
    """
    print(startup_banner)
    
    logger.info(f"--- Initializing Telegram Bot (PID: {os.getpid()}) ---")
    
    # Log system and environment information
    import platform
    import sys
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Using Telegram Bot Token: {'*' * (len(TELEGRAM_TOKEN or '') - 4)}{(TELEGRAM_TOKEN or '')[-4:] if TELEGRAM_TOKEN else 'None'}")
    logger.info(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    logger.info(f"Message cache size per chat: {MESSAGE_CACHE_SIZE}")
    logger.info(f"Default summary length: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}")
    logger.info(f"Summary command cooldown: {SUMMARY_COOLDOWN_SECONDS} seconds per user")
    logger.info(f"API timeout: {API_TIMEOUT_SECONDS} seconds")
    logger.info(f"Cache cleanup interval: {CACHE_CLEANUP_INTERVAL} seconds")
    if ADMIN_CHAT_ID:
        logger.info(f"Admin chat ID for notifications: {ADMIN_CHAT_ID}")
    else:
        logger.warning("ADMIN_CHAT_ID not set. Error notifications will not be sent via Telegram.")

    try:
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
        print("✅ Bot is now online and listening for messages!")
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    except Exception as e:
        logger.critical(f"Critical error during bot initialization or polling: {e}", exc_info=True)
        print(f"❌ Error starting bot: {e}")
        if ADMIN_CHAT_ID:
            # Try to notify admin even if we can't start polling
            try:
                import httpx
                message = f"🚨 CRITICAL: Bot failed to start!\nError: {type(e).__name__} - {e}"
                url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                httpx.post(url, json={"chat_id": ADMIN_CHAT_ID, "text": message})
            except Exception as notify_err:
                logger.error(f"Failed to notify admin about startup failure: {notify_err}")
    finally:
        # Perform cleanup
        logger.info("--- Bot polling stopped, cleaning up resources ---")
        print("👋 Bot is shutting down...")

if __name__ == "__main__":
    main()