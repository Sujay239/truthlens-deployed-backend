from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import database, models, schemas, dependencies
from typing import List
from datetime import datetime, timedelta
from sqlalchemy import func

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"]
)

@router.get("/overview", response_model=schemas.DashboardOverview)
async def get_dashboard_overview(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(dependencies.get_current_user)
):
    # 1. Calculate Stats
    total_scans = db.query(models.AnalysisLog).filter(models.AnalysisLog.user_id == current_user.id).count()
    real_scans = db.query(models.AnalysisLog).filter(
        models.AnalysisLog.user_id == current_user.id,
        models.AnalysisLog.result_label == "Real"
    ).count()
    fake_scans = db.query(models.AnalysisLog).filter(
        models.AnalysisLog.user_id == current_user.id,
        models.AnalysisLog.result_label == "Fake"  # Or "Deepfake" depending on your logic
    ).count()

    # Calculate change (dummy logic for now, or compare with last month)
    # real_change = "+8.2%" # You could query last month's count to calc this
    
    stats = [
        schemas.DashboardStats(
            title="Total Scans",
            value=f"{total_scans}",
            change="+0%", # To be implemented with time-based query
            icon_type="activity"
        ),
        schemas.DashboardStats(
            title="Real Content",
            value=f"{real_scans}",
            change="+0%",
            icon_type="check"
        ),
        schemas.DashboardStats(
            title="Fake Detected",
            value=f"{fake_scans}",
            change="+0%",
            icon_type="alert"
        )
    ]

    # 2. Chart Data (Last 7 Days)
    today = datetime.utcnow().date()
    days_map = {}
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        days_map[day.strftime("%a")] = 0

    # Aggregate counts by day
    logs_last_7_days = db.query(models.AnalysisLog).filter(
        models.AnalysisLog.user_id == current_user.id,
        models.AnalysisLog.date_created >= datetime.utcnow() - timedelta(days=7)
    ).all()

    for log in logs_last_7_days:
        day_str = log.date_created.strftime("%a")
        if day_str in days_map:
            days_map[day_str] += 1
            
    chart_data = [schemas.ChartData(name=day, scans=count) for day, count in days_map.items()]


    # 3. Pie Data
    pie_data = [
        schemas.PieData(name="Real", value=real_scans),
        schemas.PieData(name="Fake", value=fake_scans)
    ]

    # 4. Recent Activity
    recent_logs = db.query(models.AnalysisLog).filter(
        models.AnalysisLog.user_id == current_user.id
    ).order_by(models.AnalysisLog.date_created.desc()).limit(5).all()

    recent_activity = []
    for log in recent_logs:
        # Determine status color/text from label
        status = log.result_label if log.result_label else "Unknown"
        
        # Calculate relative time
        diff = datetime.utcnow() - log.date_created
        if diff.days > 0:
            date_str = f"{diff.days}d ago"
        elif diff.seconds > 3600:
            date_str = f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            date_str = f"{diff.seconds // 60}m ago"
        else:
            date_str = "Just now"

        recent_activity.append(
            schemas.RecentActivityItem(
                id=log.id,
                type=log.file_type.lower() if log.file_type else "unknown",
                name=log.filename if log.filename else "Untitled",
                status=status,
                date=date_str,
                confidence=f"{log.confidence_score:.1f}%" if log.confidence_score > 1 else f"{log.confidence_score * 100:.1f}%" if log.confidence_score else "N/A"
            )
        )

    return schemas.DashboardOverview(
        stats=stats,
        chart_data=chart_data,
        pie_data=pie_data,
        recent_activity=recent_activity
    )
