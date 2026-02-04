from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime

# --- User Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str  # Can be username or email
    password: str

class User(UserBase):
    id: int
    is_active: bool
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    avatar: Optional[str] = None

    class Config:
        from_attributes = True

class UserData(BaseModel):
    username: str
    email: str
    first_name: str
    last_name: str
    phone_number: Optional[str] = None
    avatar: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None

class UserPasswordUpdate(BaseModel):
    current_password: str
    new_password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

# --- Common Response Models ---
class BaseAnalysisResult(BaseModel):
    label: str
    confidence_score: float
    analysis_text: str

# --- Fake News ---
class FakeNewsRequest(BaseModel):
    text: str

class FakeNewsResponse(BaseAnalysisResult):
    emotional_tone: str
    source_credibility: str
    semantic_consistency: str

# --- Deepfake Image ---
# Image upload is handled via Form data / UploadFile, so we might not need a specialized Request Schema for the body itself if using fastapi.UploadFile
class ImageScanResponse(BaseAnalysisResult):
    visual_artifacts: str
    pixel_consistency: str
    metadata_analysis: str

# --- Deepfake Video ---
class VideoScanResponse(BaseAnalysisResult):
    frame_consistency: str
    audio_visual_sync: str
    blinking_patterns: str

# --- Deepfake Voice ---
class AudioScanResponse(BaseAnalysisResult):
    spectral_analysis: str
    voice_cloning_signature: str
    background_noise: str

# --- AI Text ---
class AiTextRequest(BaseModel):
    text: str

class AiTextResponse(BaseAnalysisResult):
    perplexity: str
    burstiness: str
    repetitive_patterns: str

# --- Malware ---
class MalwareUrlRequest(BaseModel):
    url: str

class MalwareResponse(BaseModel):
    label: str
    threat_score: int
    threat_level: str
    signature_match: str
    heuristic_score: str
    analysis_text: str

# --- History Log ---
class AnalysisLogResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    result_label: str
    confidence_score: float
    date_created: datetime
    file_size: Optional[str] = None
    media_url: Optional[str] = None
    analysis_summary: Optional[dict] = None
    
    class Config:
        from_attributes = True

# --- Dashboard Schemas ---
class DashboardStats(BaseModel):
    title: str
    value: str
    change: str
    icon_type: str

class ChartData(BaseModel):
    name: str
    scans: int

class PieData(BaseModel):
    name: str
    value: int

class RecentActivityItem(BaseModel):
    id: int
    type: str
    name: str
    status: str
    date: str
    confidence: str

class DashboardOverview(BaseModel):
    stats: List[DashboardStats]
    chart_data: List[ChartData]
    pie_data: List[PieData]
    recent_activity: List[RecentActivityItem]
