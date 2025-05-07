"""
Telegram Group Chat Summarizer Bot (Webhook Version)

This bot listens to messages in Telegram group chats, stores them in memory,
and generates concise summaries using Google's Gemini AI model when requested.
This version uses webhooks for deployment on platforms like Render.
"""
import logging
import os
import sys
import hashlib
import hmac
from bot import setup_bot, TELEGRAM_TOKEN, logger, notify_admin_of_error

# Web server imports
from fastapi import FastAPI, Request, Response, status, Header
from fastapi.responses import JSONResponse
import uvicorn
import json
import time
import psutil
import datetime

# --- FastAPI Setup ---
app = FastAPI(title="Telegram Bot Webhook")

# Add monitoring middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming HTTP requests."""
    start_time = time.time()
    request_id = str(time.time()).replace(".", "")[-10:]  # Simple request ID
    
    # Log basic request info
    logger.info(f"[{request_id}] Request started: {request.method} {request.url.path}")
    
    try:
        # Process the request and measure time
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(
            f"[{request_id}] Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.3f}s"
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"[{request_id}] Request failed: {request.method} {request.url.path} "
            f"- Error: {str(e)} - Time: {process_time:.3f}s",
            exc_info=True
        )
        raise

# Get port from environment variable (for Render compatibility)
PORT = int(os.environ.get("PORT", 8080))
# Get the webhook URL from environment (required for production)
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
# Get webhook path (default to /webhook)
WEBHOOK_PATH = os.environ.get("WEBHOOK_PATH", "/webhook")
# Secret token for webhook validation
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")

# --- Application Setup ---
@app.on_event("startup")
async def startup_event():
    """Initialize the bot and set the webhook on startup."""
    try:
        if not WEBHOOK_URL and not os.environ.get("DEVELOPMENT", ""):
            logger.error("WEBHOOK_URL environment variable is not set! The bot won't receive updates.")
            print("ERROR: WEBHOOK_URL environment variable is not set! The bot won't receive updates.")
            if not os.environ.get("IGNORE_WEBHOOK_ERRORS", ""):
                sys.exit(1)
        
        # Start the bot
        try:
            bot_app = setup_bot()
            logger.info("Bot application initialized successfully")
        except Exception as bot_init_error:
            logger.critical(f"Failed to initialize bot: {bot_init_error}", exc_info=True)
            print(f"❌ Failed to initialize bot: {bot_init_error}")
            sys.exit(1)
        
        # Set the webhook
        if WEBHOOK_URL:
            webhook_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
            try:
                # Include secret token if available
                webhook_params = {"url": webhook_url}
                if WEBHOOK_SECRET:
                    webhook_params["secret_token"] = WEBHOOK_SECRET
                    logger.info("Using secret token for webhook validation")
                
                await bot_app.bot.set_webhook(**webhook_params)
                logger.info(f"Webhook set to {webhook_url}")
                print(f"✅ Webhook set to {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to set webhook: {e}", exc_info=True)
                print(f"❌ Failed to set webhook: {e}")
                if not os.environ.get("IGNORE_WEBHOOK_ERRORS", ""):
                    sys.exit(1)
        
        # Store the application for later use
        app.state.bot_app = bot_app
        
    except Exception as e:
        logger.critical(f"Unexpected error during startup: {e}", exc_info=True)
        print(f"❌ Unexpected error during startup: {e}")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if hasattr(app.state, "bot_app"):
        logger.info("Shutting down bot application...")
        try:
            # Call the post_shutdown handler if it exists
            if hasattr(app.state.bot_app, "post_shutdown"):
                await app.state.bot_app.post_shutdown()
            else:
                # Fallback to direct shutdown
                await app.state.bot_app.shutdown()
            logger.info("Bot application shut down.")
        except Exception as e:
            logger.error(f"Error during bot shutdown: {e}", exc_info=True)
            print(f"❌ Error during bot shutdown: {e}")

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str = Header(None)):
    """Handle incoming webhook requests from Telegram."""
    start_time = time.time()
    
    if not hasattr(app.state, "bot_app"):
        return JSONResponse(content={"error": "Bot application not initialized"}, status_code=500)
    
    # Validate webhook secret if configured
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        logger.warning(f"Received webhook request with invalid secret token")
        return Response(status_code=status.HTTP_403_FORBIDDEN)
    
    try:
        # Try to read the request JSON
        try:
            update_data = await request.json()
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse webhook request JSON: {json_err}")
            return JSONResponse(
                content={"error": "Invalid JSON in request"},
                status_code=status.HTTP_400_BAD_REQUEST
            )
            
        logger.debug(f"Received update: {json.dumps(update_data)}")
        
        # Process the update - convert JSON to Update object
        from telegram import Update
        try:
            update_obj = Update.de_json(data=update_data, bot=app.state.bot_app.bot)
            if not update_obj:
                logger.warning(f"Failed to convert update data to Update object: {update_data}")
                return JSONResponse(
                    content={"error": "Invalid update format"},
                    status_code=status.HTTP_400_BAD_REQUEST
                )
        except Exception as update_err:
            logger.error(f"Error converting JSON to Update object: {update_err}")
            return JSONResponse(
                content={"error": "Invalid update format"},
                status_code=status.HTTP_400_BAD_REQUEST
            )
            
        # Process the update
        await app.state.bot_app.process_update(update=update_obj)
        
        # Log processing time
        processing_time = time.time() - start_time
        logger.debug(f"Update processed in {processing_time:.3f} seconds")
        
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        # Log processing time even on error
        processing_time = time.time() - start_time
        logger.error(f"Error processing update (took {processing_time:.3f} seconds): {e}", exc_info=True)
        
        # Try to notify admin about the error
        try:
            await notify_admin_of_error(app.state.bot_app, e, "webhook_handler")
        except Exception as notify_err:
            logger.error(f"Failed to notify admin about webhook error: {notify_err}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    health_data = {
        "status": "error",
        "message": "Bot not initialized",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.2.0",
    }
    
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    if hasattr(app.state, "bot_app"):
        try:
            # Check if the bot is responsive
            me = await app.state.bot_app.bot.get_me()
            
            # Check memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            health_data.update({
                "status": "ok",
                "message": "Bot is running",
                "bot_info": {
                    "username": me.username,
                    "id": me.id,
                    "is_bot": me.is_bot
                },
                "memory_usage_mb": round(memory_mb, 2),
                "uptime_seconds": int(time.time() - process.create_time())
            })
            status_code = status.HTTP_200_OK
            
        except Exception as e:
            health_data.update({
                "status": "degraded",
                "message": f"Bot check failed: {str(e)}",
                "error": str(e)
            })
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(content=health_data, status_code=status_code)

@app.get("/")
async def root():
    """Documentation endpoint that provides basic information about the bot"""
    return {
        "name": "Telegram Group Chat Summarizer Bot",
        "version": "1.2.0",
        "description": "A bot that listens to messages in Telegram group chats and generates concise summaries using Google's Gemini AI model.",
        "endpoints": {
            WEBHOOK_PATH: "The webhook endpoint for Telegram updates",
            "/health": "Health check endpoint for monitoring the bot's status"
        },
        "docs": "For more information, visit the GitHub repository"
    }

# For local development
if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=PORT,
        reload=os.environ.get("DEVELOPMENT", "").lower() == "true"
    ) 