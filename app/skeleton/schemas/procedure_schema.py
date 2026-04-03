from typing import Optional, Union
from pydantic import BaseModel


class OperationalProcedureSchema(BaseModel):
    """Schema for a single MASA operational procedure (MASA-OPS-XXX).

    Attributes:
        name: Human-readable procedure name.
        description: Plain-language description of the procedure.
        frequency: How often the procedure is executed (e.g., daily, weekly, as_needed).
        duration_minutes: Expected duration in minutes. None if variable.
        responsible: List of crew roles responsible for executing this procedure.
        scope: Module(s) or scope indicator (all_modules or list of module ids).
        systems_checked: Systems inspected or operated during this procedure.
        schedule_utc: UTC time(s) for scheduled procedures.
        schedule_day: Day of week for weekly procedures.
        trigger: Condition that triggers on-demand procedures.
        support_crew: Additional crew providing support (not primary responsible).
        director: Crew role directing the procedure (used for drills).
    """

    name: str
    description: str
    frequency: str
    duration_minutes: Optional[Union[int, str]]
    responsible: Union[list[str], str]
    scope: Union[list[str], str]
    systems_checked: Union[list[str], str]
    schedule_utc: Optional[Union[str, list[str]]] = None
    schedule_day: Optional[str] = None
    trigger: Optional[str] = None
    support_crew: Optional[list[str]] = None
    director: Optional[str] = None
