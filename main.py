import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# -------------------------------
# BASIC LOGGING
# -------------------------------
logging.basicConfig(level=logging.INFO)
print("MAIN.PY IMPORTED")

# -------------------------------
# CREATE FASTAPI APP
# -------------------------------
app = FastAPI(
    title="TruthLens AI Backend",
    version="1.0.0"
)

# -------------------------------
# CORS CONFIG
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# STATIC FILES SETUP
# -------------------------------
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="uploads"), name="static")


# -------------------------------
# VALIDATION ERROR DEBUG HANDLER
# -------------------------------
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

    error_details = exc.errors()

    for error in error_details:
        if error.get("loc") == ("body", "file") and error.get("type") == "missing":
            logging.error(
                "HINT: Send MULTIPART/FORM-DATA request with key named 'file'."
            )

    return JSONResponse(
        status_code=422,
        content={
            "detail": error_details,
            "body": str(exc),
            "hint": "Check server logs for specific hints."
        },
    )


# -------------------------------
# BASIC ROUTES
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}


@app.get("/test")
def test_route():
    return {
        "status": "success",
        "message": "Test route is working",
        "data": {
            "version": "1.0.0",
            "backend": "FastAPI"
        }
    }


# -------------------------------
# STARTUP EVENT (SAFE INITIALIZATION)
# -------------------------------
@app.on_event("startup")
async def startup_event():
    """
    All heavy operations go here.
    This prevents Render startup blocking.
    """
    print("STARTUP EVENT RUNNING...")

    try:
        # Import INSIDE startup (important!)
        from app import models
        from app.database import engine

        models.Base.metadata.create_all(bind=engine)
        print("Database initialized successfully.")

    except Exception as e:
        print(f"Database initialization skipped: {e}")

    try:
        # Import routers lazily
        from app.routers import auth, scan, history, email_test, dashboard

        app.include_router(auth.router)
        app.include_router(scan.router)
        app.include_router(history.router)
        app.include_router(email_test.router)
        app.include_router(dashboard.router)

        print("Routers loaded successfully.")

    except Exception as e:
        print(f"Router loading error: {e}")

    print("STARTUP COMPLETE ✅")


# -------------------------------
# RUN SERVER (RENDER COMPATIBLE)
# -------------------------------
if __name__ == "__main__":
    import uvicorn

    print("STARTING UVICORN...")

    # Render provides PORT automatically
    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
