"""
Main FastAPI application - Outbound Call Service
Organized and refactored for better maintainability
"""
import asyncio
import logging
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import route modules
from routes.auth_routes import router as auth_router
from routes.dashboard_routes import router as dashboard_router
from routes.user_routes import router as user_router
from routes.twilio_routes import router as twilio_router

# Import background tasks
from call_monitor import monitor_calls_and_balance

# Import startup tasks
from startup_tasks import register_google_sheets_tool

logger = logging.getLogger(__name__)

app = FastAPI(title="Outbound Call Service")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(user_router)
app.include_router(twilio_router)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting up Outbound Call Service...")
    asyncio.create_task(monitor_calls_and_balance())
    await register_google_sheets_tool()
    logger.info("✅ Outbound Call Service startup complete!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "outbound-call-service"}

# Basic page routes
from fastapi import Request
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})