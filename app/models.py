from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    hashed_password = Column(String(255))
    full_name = Column(String(255), nullable=True) # Keeping for backward compatibility or display
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone_number = Column(String(50), nullable=True)
    avatar = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    is_2fa_enabled = Column(Boolean, default=False)
    reset_token = Column(String(500), nullable=True)
    reset_token_expiry = Column(DateTime, nullable=True)

    analyses = relationship("AnalysisLog", back_populates="user")
    
    # Relationships to specific scans
    fake_news_scans = relationship("FakeNewsScan", back_populates="user")
    image_scans = relationship("ImageScan", back_populates="user")
    video_scans = relationship("VideoScan", back_populates="user")
    audio_scans = relationship("AudioScan", back_populates="user")
    ai_text_scans = relationship("AiTextScan", back_populates="user")
    malware_scans = relationship("MalwareScan", back_populates="user")

class AnalysisLog(Base):
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    filename = Column(String(255)) # Or Content Snippet
    file_type = Column(String(50)) # Image, Video, Audio, Text, URL
    date_created = Column(DateTime, default=datetime.utcnow)
    
    result_label = Column(String(50)) # Real, Fake, Suspicious, AI Generated...
    confidence_score = Column(Float)
    file_size = Column(String(50), nullable=True)
    
    media_url = Column(String(500), nullable=True) 
    
    # Generic JSON for quick retrieval in history list if needed
    analysis_summary = Column(JSON, nullable=True) 

    user = relationship("User", back_populates="analyses")

# --- Specific Scan Tables ---

class FakeNewsScan(Base):
    __tablename__ = "fake_news_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content_text = Column(Text) # Store the text analyzed
    label = Column(String(50)) # Real, Fake, Leaning Fake...
    confidence_score = Column(Float)
    emotional_tone = Column(String(255))
    source_credibility = Column(String(255))
    semantic_consistency = Column(String(255))
    analysis_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="fake_news_scans")

class ImageScan(Base):
    __tablename__ = "image_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_url = Column(String(500))
    label = Column(String(50)) # Real, Fake
    confidence_score = Column(Float)
    visual_artifacts = Column(String(255))
    pixel_consistency = Column(String(255))
    metadata_analysis = Column(String(255))
    analysis_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="image_scans")

class VideoScan(Base):
    __tablename__ = "video_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    video_url = Column(String(500))
    label = Column(String(50))
    confidence_score = Column(Float)
    frame_consistency = Column(String(255))
    audio_visual_sync = Column(String(255))
    blinking_patterns = Column(String(255))
    analysis_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    video_hash = Column(String(64), index=True, nullable=True) # SHA256 Hash
    user = relationship("User", back_populates="video_scans")

class AudioScan(Base):
    __tablename__ = "audio_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    audio_url = Column(String(500))
    label = Column(String(50))
    confidence_score = Column(Float)
    spectral_analysis = Column(String(255))
    voice_cloning_signature = Column(String(255))
    background_noise = Column(String(255))
    analysis_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="audio_scans")

class AiTextScan(Base):
    __tablename__ = "ai_text_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content_text = Column(Text)
    label = Column(String(50)) # Human Written, AI Generated
    confidence_score = Column(Float)
    perplexity = Column(String(255))
    burstiness = Column(String(255))
    repetitive_patterns = Column(String(255))
    analysis_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="ai_text_scans")

class MalwareScan(Base):
    __tablename__ = "malware_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    target = Column(String(500)) # File name or URL
    scan_type = Column(String(50)) # "Files" or "URL"
    label = Column(String(50)) # Clean, Suspicious, Malicious
    threat_score = Column(Integer)
    threat_level = Column(String(50)) # Low, Medium, High
    signature_match = Column(String(255))
    heuristic_score = Column(String(50))
    analysis_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="malware_scans")
