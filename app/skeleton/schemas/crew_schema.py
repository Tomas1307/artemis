from pydantic import BaseModel


class CrewMemberSchema(BaseModel):
    """Schema for a single MASA crew member.

    Attributes:
        name: Full legal name of the crew member.
        nationality: Nationality or nationalities.
        age: Age at mission start (Condor-7).
        assigned_module: Module identifier where this crew member is primarily stationed.
        specialization: Primary technical specializations.
        experience_years: Years of space mission experience.
        previous_missions: List of prior MASA mission identifiers.
    """

    name: str
    nationality: str
    age: int
    assigned_module: str
    specialization: str
    experience_years: int
    previous_missions: list[str]
