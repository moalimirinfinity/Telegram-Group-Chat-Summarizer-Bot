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
    """Detect the most common language in a list of messages."""
    if not messages:
        return "en"  # Default to English if no messages
    
    # Concatenate all message texts with spaces
    all_text = " ".join([msg[2] for msg in messages if msg[2] and len(msg[2]) > 5])
    
    # Return English if not enough text to detect
    if len(all_text) < 20:
        return "en"
        
    try:
        # Detect language
        detected = detect(all_text)
        return detected
    except LangDetectException:
        return "en"  # Default to English on detection error

def create_message_link(chat_id: int, message_id: int) -> str:
    """Create a direct link to a message in a Telegram chat.
    
    Note: For these links to work:
    1. The bot must be in a supergroup
    2. The chat must have a username or be a public group
    3. For private groups, links may not work for all users
    """
    # For supergroups, we need to convert the chat_id to a special format
    # Telegram uses a format where the ID needs to be modified
    if chat_id < 0:
        # Remove the negative sign and first number (usually -100)
        # But keep all digits for newer supergroups which may have a different prefix
        chat_id_str = str(abs(chat_id))
        if chat_id_str.startswith('100'):
            chat_id_str = chat_id_str[3:]  # Remove '100' prefix for supergroups
    else:
        chat_id_str = str(chat_id)
        
    return f"https://t.me/c/{chat_id_str}/{message_id}"

def add_message_links(summary: str, chat_id: int) -> str:
    """Replace message ID references with clickable links."""
    # Match patterns like [Message 123] or [Messages 123, 124, 125]
    import re
    
    # Handle single message references - [Message 123]
    def replace_single_msg(match):
        msg_id = int(match.group(1))
        return f"[Message {msg_id}]({create_message_link(chat_id, msg_id)})"
    
    # Handle multiple message references - [Messages 123, 124, 125]
    def replace_multi_msg(match):
        msg_ids_str = match.group(1)
        msg_ids = [id.strip() for id in msg_ids_str.split(',')]
        links = []
        for msg_id in msg_ids:
            try:
                msg_id_int = int(msg_id)
                links.append(f"[{msg_id}]({create_message_link(chat_id, msg_id_int)})")
            except ValueError:
                links.append(msg_id)
        return f"[Messages {', '.join(links)}]"
    
    # English pattern
    summary = re.sub(r'\[Message (\d+)\]', lambda m: replace_single_msg(m), summary)
    summary = re.sub(r'\[Messages ([\d\s,]+)\]', lambda m: replace_multi_msg(m), summary)
    
    # Spanish pattern
    summary = re.sub(r'\[Mensaje (\d+)\]', lambda m: replace_single_msg(m), summary)
    summary = re.sub(r'\[Mensajes ([\d\s,]+)\]', lambda m: replace_multi_msg(m), summary)
    
    # Russian pattern
    summary = re.sub(r'\[Сообщение (\d+)\]', lambda m: replace_single_msg(m), summary)
    summary = re.sub(r'\[Сообщения ([\d\s,]+)\]', lambda m: replace_multi_msg(m), summary)
    
    # French pattern
    summary = re.sub(r'\[Message (\d+)\]', lambda m: replace_single_msg(m), summary)
    summary = re.sub(r'\[Messages ([\d\s,]+)\]', lambda m: replace_multi_msg(m), summary)
    
    # German pattern
    summary = re.sub(r'\[Nachricht (\d+)\]', lambda m: replace_single_msg(m), summary)
    summary = re.sub(r'\[Nachrichten ([\d\s,]+)\]', lambda m: replace_multi_msg(m), summary)
    
    return summary

# --- Telegram Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start or /help command."""
    if not update.message:
        return

    # Get user locale from Telegram
    user_lang_code = update.effective_user.language_code if update.effective_user else None
    
    # Fallback to 'en' if not available or not supported
    lang = user_lang_code if user_lang_code in ["en", "es", "ru", "fr", "de"] else "en"
    
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
        "es": (
            "¡Hola! Soy un bot diseñado para resumir mensajes recientes en este grupo.\n\n"
            f"Usa el comando `/{COMMAND_NAME} [N]` donde `N` es el número de mensajes recientes que quieres resumir. "
            f"(Predeterminado: {DEFAULT_SUMMARY_MESSAGES}, Máx: {MAX_SUMMARY_MESSAGES}).\n"
            f"Ejemplo: `/{COMMAND_NAME} 50`\n\n"
            f"Uso el modelo `{GEMINI_MODEL_NAME}` para generar resúmenes.\n"
            f"Hay un tiempo de espera de {SUMMARY_COOLDOWN_SECONDS} segundos por usuario para este comando para prevenir spam.\n\n"
            "**Importante:** Para que pueda ver mensajes y resumirlos, el modo 'Privacidad de Grupo' debe estar **desactivado** en mis ajustes. "
            "Puedes gestionarlo a través de @BotFather (`/mybots` -> selecciona bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
        ),
        "ru": (
            "Привет! Я бот, созданный для обобщения недавних сообщений в этой группе.\n\n"
            f"Используйте команду `/{COMMAND_NAME} [N]`, где `N` - количество последних сообщений, которые вы хотите обобщить. "
            f"(По умолчанию: {DEFAULT_SUMMARY_MESSAGES}, Макс: {MAX_SUMMARY_MESSAGES}).\n"
            f"Пример: `/{COMMAND_NAME} 50`\n\n"
            f"Я использую модель `{GEMINI_MODEL_NAME}` для создания резюме.\n"
            f"Есть {SUMMARY_COOLDOWN_SECONDS}-секундная задержка на каждого пользователя для этой команды, чтобы предотвратить спам.\n\n"
            "**Важно:** Чтобы я мог видеть сообщения и делать резюме, режим 'Приватность группы' должен быть **отключен** в моих настройках. "
            "Вы можете управлять этим через @BotFather (`/mybots` -> выберите бота -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
        ),
        "fr": (
            "Bonjour ! Je suis un bot conçu pour résumer les messages récents dans ce groupe.\n\n"
            f"Utilisez la commande `/{COMMAND_NAME} [N]` où `N` est le nombre de messages récents que vous souhaitez résumer. "
            f"(Par défaut : {DEFAULT_SUMMARY_MESSAGES}, Max : {MAX_SUMMARY_MESSAGES}).\n"
            f"Exemple : `/{COMMAND_NAME} 50`\n\n"
            f"J'utilise le modèle `{GEMINI_MODEL_NAME}` pour générer des résumés.\n"
            f"Il y a un temps de recharge de {SUMMARY_COOLDOWN_SECONDS} secondes par utilisateur pour cette commande afin d'éviter le spam.\n\n"
            "**Important :** Pour que je puisse voir les messages et les résumer, le mode 'Confidentialité des groupes' doit être **désactivé** dans mes paramètres. "
            "Vous pouvez gérer cela via @BotFather (`/mybots` -> sélectionnez le bot -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
        ),
        "de": (
            "Hallo! Ich bin ein Bot, der entwickelt wurde, um aktuelle Nachrichten in dieser Gruppe zusammenzufassen.\n\n"
            f"Verwenden Sie den Befehl `/{COMMAND_NAME} [N]`, wobei `N` die Anzahl der aktuellen Nachrichten ist, die Sie zusammenfassen möchten. "
            f"(Standard: {DEFAULT_SUMMARY_MESSAGES}, Max: {MAX_SUMMARY_MESSAGES}).\n"
            f"Beispiel: `/{COMMAND_NAME} 50`\n\n"
            f"Ich verwende das Modell `{GEMINI_MODEL_NAME}` zum Generieren von Zusammenfassungen.\n"
            f"Es gibt eine {SUMMARY_COOLDOWN_SECONDS}-Sekunden Abklingzeit pro Benutzer für diesen Befehl, um Spam zu verhindern.\n\n"
            "**Wichtig:** Damit ich Nachrichten sehen und zusammenfassen kann, muss der 'Gruppendatenschutz'-Modus in meinen Einstellungen **deaktiviert** sein. "
            "Sie können dies über @BotFather verwalten (`/mybots` -> Bot auswählen -> `Bot Settings` -> `Group Privacy` -> `Turn off`)."
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

    # 3. Parse arguments
    num_messages = DEFAULT_SUMMARY_MESSAGES
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
                num_messages = DEFAULT_SUMMARY_MESSAGES
        except (ValueError, IndexError):
             await message.reply_text(
                 f"Invalid number format. Using default {DEFAULT_SUMMARY_MESSAGES}. "
                 f"Usage: `/{COMMAND_NAME} [number]` (e.g., `/{COMMAND_NAME} 50`).",
                  parse_mode=constants.ParseMode.MARKDOWN,
             )
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
                    "Provide a *concise, neutral, and well-structured* summary of the following messages. "
                    "Focus on key discussion points, decisions made, questions asked, and any action items mentioned.\n"
                    "Format the summary clearly using bullet points or numbered lists for better readability.\n"
                    "Organize the content by topics or themes and use formatting to create a structured summary.\n"
                    "When referencing specific messages, use the message ID in a reference like '[Message 123]' at the end of the relevant point.\n"
                    "If multiple messages discuss the same point, group them like '[Messages 123, 124, 125]'.\n"
                    "Keep these references minimal and only for important points that users might want to find in the original chat.",
            "summary_request": "Concise Summary:",
            "processing_message": f"⏳ Fetching and summarizing the last {actual_count} cached messages using AI... please wait.",
            "error_message": "❌ Oops! Something went wrong while generating the summary. Please try again later. If the problem persists, contact the bot admin.",
            "summary_header": f"**✨ SUMMARY OF RECENT MESSAGES ✨**\n\n",
            "summary_footer": "\n\n*Note: Message links work in supergroups and public groups. In private groups, links may not work for all users.*"
        },
        "es": {
            "intro": "Eres un asistente útil encargado de resumir conversaciones de grupo de Telegram.\n"
                    "Proporciona un resumen *conciso, neutral y bien estructurado* de los siguientes mensajes. "
                    "Céntrate en los puntos clave de discusión, decisiones tomadas, preguntas realizadas y cualquier elemento de acción mencionado.\n"
                    "Formatea el resumen claramente usando viñetas o listas numeradas para mejor legibilidad.\n"
                    "Organiza el contenido por temas y usa formato para crear un resumen estructurado.\n"
                    "Cuando hagas referencia a mensajes específicos, usa el ID del mensaje en formato '[Mensaje 123]' al final del punto relevante.\n"
                    "Si varios mensajes discuten el mismo punto, agrúpalos como '[Mensajes 123, 124, 125]'.\n"
                    "Mantén estas referencias al mínimo y solo para puntos importantes que los usuarios quieran encontrar en el chat original.",
            "summary_request": "Resumen conciso:",
            "processing_message": f"⏳ Obteniendo y resumiendo los últimos {actual_count} mensajes usando IA... por favor espera.",
            "error_message": "❌ ¡Ups! Algo salió mal al generar el resumen. Por favor, inténtalo de nuevo más tarde. Si el problema persiste, contacta al administrador del bot.",
            "summary_header": f"**✨ RESUMEN DE MENSAJES RECIENTES ✨**\n\n",
            "summary_footer": "\n\n*Nota: Los enlaces a mensajes funcionan en supergrupos y grupos públicos. En grupos privados, es posible que los enlaces no funcionen para todos los usuarios.*"
        },
        "ru": {
            "intro": "Вы - полезный ассистент, которому поручено обобщать групповые чаты Telegram.\n"
                    "Предоставьте *краткое, нейтральное и хорошо структурированное* резюме следующих сообщений. "
                    "Сосредоточьтесь на ключевых моментах обсуждения, принятых решениях, заданных вопросах и любых упомянутых действиях.\n"
                    "Четко форматируйте резюме, используя маркеры или нумерованные списки для лучшей читаемости.\n"
                    "Организуйте содержание по темам и используйте форматирование для создания структурированного резюме.\n"
                    "При ссылке на конкретные сообщения используйте ID сообщения в формате '[Сообщение 123]' в конце соответствующего пункта.\n"
                    "Если несколько сообщений обсуждают один и тот же вопрос, сгруппируйте их как '[Сообщения 123, 124, 125]'.\n"
                    "Сохраняйте эти ссылки минимальными и только для важных пунктов, которые пользователи могут захотеть найти в оригинальном чате.",
            "summary_request": "Краткое резюме:",
            "processing_message": f"⏳ Получение и составление резюме последних {actual_count} сообщений с использованием ИИ... пожалуйста, подождите.",
            "error_message": "❌ Упс! Что-то пошло не так при генерации резюме. Пожалуйста, повторите попытку позже. Если проблема не исчезнет, обратитесь к администратору бота.",
            "summary_header": f"**✨ РЕЗЮМЕ НЕДАВНИХ СООБЩЕНИЙ ✨**\n\n",
            "summary_footer": "\n\n*Примечание: Ссылки на сообщения работают в супергруппах и публичных группах. В приватных группах ссылки могут работать не для всех пользователей.*"
        },
        "fr": {
            "intro": "Vous êtes un assistant utile chargé de résumer les conversations de groupe Telegram.\n"
                    "Fournissez un résumé *concis, neutre et bien structuré* des messages suivants. "
                    "Concentrez-vous sur les points clés de discussion, les décisions prises, les questions posées et toute action mentionnée.\n"
                    "Formatez clairement le résumé en utilisant des puces ou des listes numérotées pour une meilleure lisibilité.\n"
                    "Organisez le contenu par thèmes et utilisez le formatage pour créer un résumé structuré.\n"
                    "Lorsque vous faites référence à des messages spécifiques, utilisez l'ID du message au format '[Message 123]' à la fin du point concerné.\n"
                    "Si plusieurs messages abordent le même sujet, regroupez-les comme '[Messages 123, 124, 125]'.\n"
                    "Gardez ces références au minimum et uniquement pour les points importants que les utilisateurs pourraient vouloir retrouver dans le chat original.",
            "summary_request": "Résumé concis:",
            "processing_message": f"⏳ Récupération et résumé des {actual_count} derniers messages avec IA... veuillez patienter.",
            "error_message": "❌ Oups! Une erreur s'est produite lors de la génération du résumé. Veuillez réessayer plus tard. Si le problème persiste, contactez l'administrateur du bot.",
            "summary_header": f"**✨ RÉSUMÉ DES MESSAGES RÉCENTS ✨**\n\n",
            "summary_footer": "\n\n*Remarque: Les liens vers les messages fonctionnent dans les supergroupes et les groupes publics. Dans les groupes privés, les liens peuvent ne pas fonctionner pour tous les utilisateurs.*"
        },
        "de": {
            "intro": "Sie sind ein hilfreicher Assistent, der Telegram-Gruppenchats zusammenfasst.\n"
                    "Geben Sie eine *präzise, neutrale und gut strukturierte* Zusammenfassung der folgenden Nachrichten. "
                    "Konzentrieren Sie sich auf wichtige Diskussionspunkte, getroffene Entscheidungen, gestellte Fragen und erwähnte Aktionspunkte.\n"
                    "Formatieren Sie die Zusammenfassung übersichtlich mit Aufzählungspunkten oder nummerierten Listen für bessere Lesbarkeit.\n"
                    "Organisieren Sie den Inhalt nach Themen und verwenden Sie Formatierung für eine strukturierte Zusammenfassung.\n"
                    "Bei Verweisen auf bestimmte Nachrichten verwenden Sie die Nachrichten-ID im Format '[Nachricht 123]' am Ende des entsprechenden Punktes.\n"
                    "Wenn mehrere Nachrichten dasselbe Thema behandeln, gruppieren Sie sie wie '[Nachrichten 123, 124, 125]'.\n"
                    "Halten Sie diese Verweise minimal und nur für wichtige Punkte, die Benutzer im ursprünglichen Chat finden möchten.",
            "summary_request": "Präzise Zusammenfassung:",
            "processing_message": f"⏳ Abrufen und Zusammenfassen der letzten {actual_count} Nachrichten mit KI... bitte warten.",
            "error_message": "❌ Hoppla! Beim Erstellen der Zusammenfassung ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut. Wenn das Problem weiterhin besteht, kontaktieren Sie den Bot-Administrator.",
            "summary_header": f"**✨ ZUSAMMENFASSUNG DER LETZTEN NACHRICHTEN ✨**\n\n",
            "summary_footer": "\n\n*Hinweis: Nachrichtenlinks funktionieren in Supergruppen und öffentlichen Gruppen. In privaten Gruppen funktionieren Links möglicherweise nicht für alle Benutzer.*"
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

        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=generation_config,
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
            summary_text = add_message_links(summary_text, chat_id)
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