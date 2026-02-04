from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import os

from app import models
from app.database import engine
from app.routers import auth, scan, history, email_test, dashboard
from app.ml.bert_classifier import get_model_and_tokenizer

# -----------------------------
# Database setup
# -----------------------------
models.Base.metadata.create_all(bind=engine)

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="TruthLens AI Backend",
    version="1.0.0"
)

# -----------------------------
# CORS configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Static files
# -----------------------------
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# -----------------------------
# Exception handler (422 debug)
# -----------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error("Validation Error encountered!")
    logging.error(f"URL: {request.url}")
    logging.error(f"Headers: {request.headers}")

    try:
        body = await request.body()
        logging.error(f"Body: {body.decode('utf-8', errors='ignore')}")
    except Exception:
        pass

    logging.error(f"Errors: {exc}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "hint": "Check server logs for validation hints"
        },
    )

# -----------------------------
# Health check routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.get("/test")
def test_route():
    return {
        "status": "success",
        "backend": "FastAPI",
        "version": "1.0.0"
    }

# -----------------------------
# Routers
# -----------------------------
app.include_router(auth.router)
app.include_router(scan.router)
app.include_router(history.router)
app.include_router(email_test.router)
app.include_router(dashboard.router)

# -----------------------------
# Startup event (ML preload)
# -----------------------------
@app.on_event("startup")
async def startup_event():
    print("Pre-loading BERT model...")
    get_model_and_tokenizer()
    print("BERT model loaded successfully!")
