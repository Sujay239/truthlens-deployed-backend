from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import models, schemas, database, dependencies
from typing import List

router = APIRouter(
    prefix="/history",
    tags=["History"]
)

@router.get("/", response_model=List[schemas.AnalysisLogResponse])
def get_user_history(db: Session = Depends(database.get_db), current_user: models.User = Depends(dependencies.get_current_user)):
    logs = db.query(models.AnalysisLog).filter(models.AnalysisLog.user_id == current_user.id).order_by(models.AnalysisLog.date_created.desc()).all()
    return logs
