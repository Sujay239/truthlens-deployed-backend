from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from datetime import timedelta
from .. import models, schemas, utils, database, dependencies
from app.email_utils import send_email
from app.email_templates import get_password_reset_template, get_new_account_admin_notification_template
import uuid
from datetime import datetime
from fastapi import BackgroundTasks, UploadFile, File
import shutil
import os

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# Google Auth Config
GOOGLE_CLIENT_ID = "469032517353-n3pg2fh1gkupkbjoqfsr1anbcjqqt21b.apps.googleusercontent.com"

# GitHub Auth Config
GITHUB_CLIENT_ID = "Ov23liDMQfI42XVPRzpE"
GITHUB_CLIENT_SECRET = "3af19222fe03da3a0ad2eeb0a99dd8b272680320"

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import secrets
import string
import requests

class GoogleLoginRequest(schemas.BaseModel):
    token: str

class GithubLoginRequest(schemas.BaseModel):
    code: str

@router.post("/github-login", response_model=schemas.Token)
def github_login(request: GithubLoginRequest, db: Session = Depends(database.get_db)):
    try:
        # 1. Exchange code for access token
        token_url = "https://github.com/login/oauth/access_token"
        headers = {"Accept": "application/json"}
        data = {
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": request.code
        }
        
        response = requests.post(token_url, headers=headers, json=data)
        if response.status_code != 200:
             raise HTTPException(status_code=400, detail="Failed to retrieve access token from GitHub")
             
        token_data = response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
             error_desc = token_data.get("error_description", "Unknown error")
             raise HTTPException(status_code=400, detail=f"GitHub Error: {error_desc}")

        # 2. user info
        user_url = "https://api.github.com/user"
        auth_headers = {"Authorization": f"token {access_token}"}
        user_response = requests.get(user_url, headers=auth_headers)
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve user info from GitHub")
            
        user_info = user_response.json()
        
        # 3. Handle Email (GitHub emails can be private)
        email = user_info.get("email")
        if not email:
            # Fetch emails manually
            emails_url = "https://api.github.com/user/emails"
            emails_response = requests.get(emails_url, headers=auth_headers)
            if emails_response.status_code == 200:
                emails = emails_response.json()
                # Find primary verified email
                for e in emails:
                    if e.get("primary") and e.get("verified"):
                        email = e.get("email")
                        break
                # Fallback to any verified email
                if not email:
                     for e in emails:
                        if e.get("verified"):
                            email = e.get("email")
                            break
                            
        if not email:
            raise HTTPException(status_code=400, detail="No verified email found for this GitHub account")

        # 4. Process User
        username = user_info.get("login")
        avatar = user_info.get("avatar_url")
        name = user_info.get("name") or username
        
        name_parts = name.split()
        first_name = name_parts[0] if name_parts else ""
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        # Check in DB
        user = db.query(models.User).filter(models.User.email == email.lower()).first()
        
        if not user:
            # Check for username collision and handle it
            if db.query(models.User).filter(models.User.username == username).first():
                username = f"{username}{secrets.randbelow(1000)}"
                
            alphabet = string.ascii_letters + string.digits
            password = ''.join(secrets.choice(alphabet) for i in range(16))
            hashed_password = utils.get_password_hash(password)
            
            user = models.User(
                email=email.lower(),
                username=username,
                hashed_password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                full_name=name,
                avatar=avatar,
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
        access_token_expires = timedelta(minutes=utils.ACCESS_TOKEN_EXPIRE_MINUTES)
        jwt_token = utils.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": jwt_token, "token_type": "bearer"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"GitHub Login Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GitHub login failed: {str(e)}")

@router.post("/google-login", response_model=schemas.Token)
def google_login(request: GoogleLoginRequest, db: Session = Depends(database.get_db)):
    try:
        # Try fetching user info with the token (treating it as an Access Token)
        # This is compatible with useGoogleLogin hook in React
        response = google_requests.requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {request.token}'}
        )
        
        email = None
        picture = None
        first_name = ""
        last_name = ""
        
        if response.status_code == 200:
            user_info = response.json()
            email = user_info.get('email')
            first_name = user_info.get('given_name', '')
            last_name = user_info.get('family_name', '')
            picture = user_info.get('picture')
        else:
            # If fetching fails, try verifying it as an ID Token (fallback)
            try:
                id_info = id_token.verify_oauth2_token(
                    request.token, 
                    google_requests.Request(), 
                    GOOGLE_CLIENT_ID
                )
                email = id_info.get('email')
                first_name = id_info.get('given_name', '')
                last_name = id_info.get('family_name', '')
                picture = id_info.get('picture')
            except ValueError:
                 raise HTTPException(status_code=401, detail="Invalid Google token")

        if not email:
            raise HTTPException(status_code=400, detail="Email not found in Google token")

        # Check if user exists
        user = db.query(models.User).filter(models.User.email == email.lower()).first()
        
        if not user:
            # Register new user
            username = email.split('@')[0]
            # Ensure username is unique
            if db.query(models.User).filter(models.User.username == username).first():
                username = f"{username}{secrets.randbelow(1000)}"
            
            # Generate random password
            alphabet = string.ascii_letters + string.digits
            password = ''.join(secrets.choice(alphabet) for i in range(16))
            hashed_password = utils.get_password_hash(password)
            
            full_name = f"{first_name} {last_name}".strip()
            
            user = models.User(
                email=email.lower(),
                username=username,
                hashed_password=hashed_password,
                first_name=first_name,
                last_name=last_name,
                full_name=full_name,
                avatar=picture,
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
        # Create access token
        access_token_expires = timedelta(minutes=utils.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = utils.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}

    except ValueError as e:
        # Invalid token
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google login failed: {str(e)}")

@router.get("/myData", response_model=schemas.UserData)
def get_my_data(current_user: models.User = Depends(dependencies.get_current_user)):
    return {
        "email": current_user.email,
        "username": current_user.username,
        "first_name": current_user.first_name if current_user.first_name else "",
        "last_name": current_user.last_name if current_user.last_name else "",
        "phone_number": current_user.phone_number,
        "avatar": current_user.avatar
    }

@router.put("/me", response_model=schemas.User)
def update_user_profile(
    user_update: schemas.UserUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    if user_update.first_name is not None:
        current_user.first_name = user_update.first_name
    
    if user_update.last_name is not None:
        current_user.last_name = user_update.last_name
        
    # Maintain full_name for backward compatibility
    # Ensure none values are empty strings when constructing full name
    fn = current_user.first_name if current_user.first_name else ""
    ln = current_user.last_name if current_user.last_name else ""
    current_user.full_name = f"{fn} {ln}".strip()
    
    if user_update.phone_number is not None:
        current_user.phone_number = user_update.phone_number
        
    db.commit()
    db.refresh(current_user)
    return current_user

@router.post("/upload-avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # Ensure directory exists
    UPLOAD_DIR = "uploads/avatars"
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        
    # Create valid filename
    file_ext = file.filename.split(".")[-1]
    filename = f"{current_user.id}_{int(datetime.utcnow().timestamp())}.{file_ext}"
    file_path = f"{UPLOAD_DIR}/{filename}"
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Update User DB
    # The URL will be /static/avatars/filename
    avatar_url = f"http://localhost:8000/uploads/avatars/{filename}"
    current_user.avatar = avatar_url
    db.commit()
    db.refresh(current_user)
    
    return {"message": "Avatar uploaded successfully", "avatar_url": avatar_url}

@router.put("/change-password")
def change_password(
    password_update: schemas.UserPasswordUpdate,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    if not utils.verify_password(password_update.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password")
        
    current_user.hashed_password = utils.get_password_hash(password_update.new_password)
    db.commit()
    
    return {"message": "Password updated successfully"}

@router.post("/register", response_model=schemas.User)
def register(
    user: schemas.UserCreate, 
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(database.get_db)
):
    # Normalize email to lowercase
    user.email = user.email.lower()
    
    # DEBUG LOGGING
    print(f"DEBUG: Registering user: {user.username}")
    print(f"DEBUG: Received Data: {user.dict()}")
    print(f"DEBUG: first_name={user.first_name}, last_name={user.last_name}")

    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = db.query(models.User).filter(models.User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = utils.get_password_hash(user.password)
    
    # Create full name for fallback
    full_name = f"{user.first_name} {user.last_name}".strip() if user.first_name or user.last_name else None
    
    db_user = models.User(
        email=user.email, 
        username=user.username, 
        hashed_password=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Send Welcome Email
    from app.email_templates import get_welcome_email_template
    subject = "Welcome to TruthLens AI! 🛡️"
    body = get_welcome_email_template(user.username)
    
    try:
        background_tasks.add_task(send_email, subject, [user.email], body)
    except Exception as e:
        print(f"Failed to send welcome email: {e}")

    # Send Admin Notification
    try:
        user_agent = request.headers.get('user-agent', 'Unknown')
        
        # Simple parsing
        platform = "Unknown"
        if "Windows" in user_agent: platform = "Windows"
        elif "Mac" in user_agent: platform = "MacOS"
        elif "Linux" in user_agent: platform = "Linux"
        elif "Android" in user_agent: platform = "Android"
        elif "iPhone" in user_agent or "iPad" in user_agent: platform = "iOS"
        
        browser = "Unknown"
        if "Edg" in user_agent: browser = "Edge"
        elif "Chrome" in user_agent: browser = "Chrome" 
        elif "Firefox" in user_agent: browser = "Firefox"
        elif "Safari" in user_agent: browser = "Safari"
        
        # IST Time (UTC + 5:30)
        ist_offset = timedelta(hours=5, minutes=30)
        ist_now = datetime.utcnow() + ist_offset
        
        admin_subject = f"New Account: {user.username}"
        user_info = {
            "username": user.username,
            "email": user.email,
            "full_name": full_name or "N/A",
            "platform": platform,
            "browser": f"{browser} ({user_agent})",
            "time": ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
        }
        admin_body = get_new_account_admin_notification_template(user_info)
        background_tasks.add_task(send_email, admin_subject, ["sujaykumarkotal8520@gmail.com"], admin_body)
    except Exception as e:
        print(f"Failed to send admin notification: {e}")

    return db_user

@router.post("/login", response_model=schemas.Token)
def login(user_credentials: schemas.UserLogin, db: Session = Depends(database.get_db)):
    # Simple JSON login
    user = db.query(models.User).filter(models.User.username == user_credentials.username).first()
    if not user:
        # Check if they used email instead (normalize to lowercase)
        user = db.query(models.User).filter(models.User.email == user_credentials.username.lower()).first()
    
    if not user or not utils.verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = utils.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
def logout():
    return {"message": "Logged out successfully"}

@router.post("/forgot-password")
async def forgot_password(
    request: schemas.ForgotPasswordRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db)
):
    user = db.query(models.User).filter(models.User.email == request.email.lower()).first()
    if not user:
        # Don't reveal if email exists, just return success
        return {"message": "If the email exists, a reset link has been sent."}

    # Generate Token
    token = str(uuid.uuid4())
    user.reset_token = token
    user.reset_token_expiry = datetime.utcnow() + timedelta(minutes=15)
    db.commit()

    # Send Email
    reset_link = f"http://localhost:5173/auth/forgot-password?token={token}"
    subject = "Reset Your Password - TruthLens AI"
    body = get_password_reset_template(reset_link)
    
    background_tasks.add_task(send_email, subject, [user.email], body)
    
    return {"message": "If the email exists, a reset link has been sent."}

@router.post("/reset-password")
def reset_password(
    request: schemas.ResetPasswordRequest, 
    db: Session = Depends(database.get_db)
):
    user = db.query(models.User).filter(models.User.reset_token == request.token).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")
    
    if user.reset_token_expiry < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Token has expired")
    
    # Update Password
    user.hashed_password = utils.get_password_hash(request.new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    db.commit()
    
    return {"message": "Password updated successfully"}
