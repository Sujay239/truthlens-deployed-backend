from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import models
from app.database import engine
from app.routers import auth, scan, history, email_test, dashboard

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="TruthLens AI Backend",
    version="1.0.0"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount Static Files
from fastapi.staticfiles import StaticFiles
import os

if not os.path.exists("uploads"):
    os.makedirs("uploads")
    
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# DEBUG: Exception Handler for 422 Errors
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logging.error(f"Validation Error encountered!")
    logging.error(f"URL: {request.url}")
    logging.error(f"Headers: {request.headers}")
    try:
        body = await request.body()
        logging.error(f"Body: {body.decode('utf-8', errors='ignore')}")
    except:
        pass
    
    logging.error(f"Errors: {exc}")
    
    # helper for file upload errors
    error_details = exc.errors()
    for error in error_details:
        if error.get("loc") == ("body", "file") and error.get("type") == "missing":
            logging.error("HINT: The 'file' field is missing. Ensure you are sending a MULTIPART/FORM-DATA request with a key named 'file'.")
            
    return JSONResponse(
        status_code=422,
        content={"detail": error_details, "body": str(exc), "hint": "Check server logs for specific hints."},
    )

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

# Include the Auth Router
app.include_router(auth.router)
app.include_router(scan.router)
app.include_router(history.router)
app.include_router(email_test.router)
app.include_router(dashboard.router)

from app.ml.bert_classifier import get_model_and_tokenizer

# @app.on_event("startup")
# async def startup_event():
#     print("Pre-loading BERT model... Please wait.")
#     get_model_and_tokenizer()
#     print("BERT model loaded and ready!")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
