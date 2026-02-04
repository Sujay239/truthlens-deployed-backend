from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.email_utils import send_email
from pydantic import BaseModel, EmailStr

router = APIRouter(
    prefix="/email",
    tags=["Email"]
)

class EmailRequest(BaseModel):
    email: EmailStr

@router.post("/send-test")
async def send_test_email(request: EmailRequest, background_tasks: BackgroundTasks):
    """
    Send a test email to the provided address.
    """
    subject = "Test Email from TruthLens AI"
    body = """
    <h1>Test Email</h1>
    <p>This is a test email sent from the TruthLens AI Backend.</p>
    <p>If you received this, the SMTP configuration is working!</p>
    """
    
    try:
        # Sending email in background to avoid blocking
        background_tasks.add_task(send_email, subject, [request.email], body)
        return {"message": "Test email queued for sending."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
