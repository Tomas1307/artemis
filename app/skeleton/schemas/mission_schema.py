from typing import Optional
from pydantic import BaseModel


class MissionSchema(BaseModel):
    """Schema for a past or current MASA mission (Cóndor-N).

    Attributes:
        name: Human-readable mission name.
        type: Mission classification (e.g., crewed, cargo, emergency_repair).
        start_date: ISO 8601 mission start date.
        end_date: ISO 8601 mission end date. None if mission is ongoing.
        duration_days: Mission duration in days. None if ongoing.
        crew: List of crew role identifiers who participated.
        objective: Primary mission objective.
        outcome: Mission result (success, in_progress, partial_success, failure).
        notes: Additional context, anomalies, or lessons learned.
    """

    name: str
    type: str
    start_date: str
    end_date: Optional[str]
    duration_days: Optional[int]
    crew: list[str]
    objective: str
    outcome: str
    notes: str
