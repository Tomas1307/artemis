from pydantic import BaseModel
from app.skeleton.schemas.module_schema import ModuleSchema
from app.skeleton.schemas.protocol_schema import SecurityProtocolSchema
from app.skeleton.schemas.procedure_schema import OperationalProcedureSchema
from app.skeleton.schemas.crew_schema import CrewMemberSchema
from app.skeleton.schemas.mission_schema import MissionSchema


class StationMetaSchema(BaseModel):
    """Metadata for the Kuntur Station and MASA agency.

    Attributes:
        station_name: Official name of the space station.
        orbit: Orbital regime description.
        altitude_km: Nominal orbital altitude in kilometers.
        agency: Agency acronym.
        agency_full: Full agency name.
        founded: Year the agency was founded.
        headquarters: Physical headquarters location.
        current_mission: Identifier of the currently active mission.
        mission_start: ISO 8601 start date of the current mission.
    """

    station_name: str
    orbit: str
    altitude_km: int
    agency: str
    agency_full: str
    founded: int
    headquarters: str
    current_mission: str
    mission_start: str


class SkeletonSchema(BaseModel):
    """Root schema for the full MASA universe skeleton.

    This is the single source of truth consumed by all generation phases.
    All question gold standards and document injected values derive from here.

    Attributes:
        meta: Station and agency metadata.
        modules: Mapping of module identifier to module specification.
        security_protocols: Mapping of protocol ID to protocol definition.
        operational_procedures: Mapping of procedure ID to procedure definition.
        crew: Mapping of crew role identifier to crew member profile.
        missions: Mapping of mission identifier to mission record.
    """

    meta: StationMetaSchema
    modules: dict[str, ModuleSchema]
    security_protocols: dict[str, SecurityProtocolSchema]
    operational_procedures: dict[str, OperationalProcedureSchema]
    crew: dict[str, CrewMemberSchema]
    missions: dict[str, MissionSchema]
