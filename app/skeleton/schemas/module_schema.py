from typing import Optional
from pydantic import BaseModel, Field


class TemperatureThresholdSchema(BaseModel):
    """Temperature alert thresholds for a module.

    Attributes:
        unit: Unit of measurement, always celsius.
        normal_min: Lower bound of normal operating range.
        normal_max: Upper bound of normal operating range.
        low_alert_max: Upper bound of low-severity alert band.
        medium_alert_max: Upper bound of medium-severity alert band.
        high_alert_max: Upper bound of high-severity alert band.
        critical_above: Value above which severity becomes critical.
    """

    unit: str
    normal_min: float
    normal_max: float
    low_alert_max: float
    medium_alert_max: float
    high_alert_max: float
    critical_above: float


class PressureThresholdSchema(BaseModel):
    """Pressure alert thresholds for a module.

    Attributes:
        unit: Unit of measurement, always kPa.
        normal_min: Lower bound of normal operating range.
        normal_max: Upper bound of normal operating range.
        low_alert_min: Lower bound triggering low-severity alert.
        medium_alert_min: Lower bound triggering medium-severity alert.
        high_alert_min: Lower bound triggering high-severity alert.
        critical_below: Value below which severity becomes critical.
    """

    unit: str
    normal_min: float
    normal_max: float
    low_alert_min: float
    medium_alert_min: float
    high_alert_min: float
    critical_below: float


class OxygenThresholdSchema(BaseModel):
    """Oxygen concentration alert thresholds for a module.

    Attributes:
        unit: Unit of measurement, always percent.
        normal_min: Lower bound of normal O2 range.
        normal_max: Upper bound of normal O2 range.
        low_alert_min: Lower bound triggering low-severity alert.
        medium_alert_min: Lower bound triggering medium-severity alert.
        high_alert_min: Lower bound triggering high-severity alert.
        critical_below: Value below which severity becomes critical.
    """

    unit: str
    normal_min: float
    normal_max: float
    low_alert_min: float
    medium_alert_min: float
    high_alert_min: float
    critical_below: float


class RadiationThresholdSchema(BaseModel):
    """Radiation alert thresholds for a module.

    Attributes:
        unit: Unit of measurement, always mSv_per_hour.
        normal_max: Upper bound of normal radiation level.
        low_alert_max: Upper bound of low-severity alert band.
        medium_alert_max: Upper bound of medium-severity alert band.
        high_alert_max: Upper bound of high-severity alert band.
        critical_above: Value above which severity becomes critical.
    """

    unit: str
    normal_max: float
    low_alert_max: float
    medium_alert_max: float
    high_alert_max: float
    critical_above: float


class HumidityThresholdSchema(BaseModel):
    """Humidity alert thresholds for a module.

    Humidity has two-sided bands (too low and too high are both alerts).

    Attributes:
        unit: Unit of measurement, always percent.
        normal_min: Lower bound of normal humidity range.
        normal_max: Upper bound of normal humidity range.
        critical_below: Value below which severity becomes critical.
        critical_above: Value above which severity becomes critical.
    """

    unit: str
    normal_min: int
    normal_max: int
    low_alert_band: list[int]
    low_alert_high_band: list[int]
    medium_alert_band: list[int]
    medium_alert_high_band: list[int]
    high_alert_band: list[int]
    high_alert_high_band: list[int]
    critical_below: int
    critical_above: int


class PowerThresholdSchema(BaseModel):
    """Power consumption alert thresholds for a module.

    Attributes:
        unit: Unit of measurement, always percent_rated_capacity.
        normal_max: Upper bound of normal power consumption.
        low_alert_max: Upper bound of low-severity alert band.
        medium_alert_max: Upper bound of medium-severity alert band.
        high_alert_max: Upper bound of high-severity alert band.
        critical_above: Value above which severity becomes critical.
    """

    unit: str
    normal_max: int
    low_alert_max: int
    medium_alert_max: int
    high_alert_max: int
    critical_above: int


class TelemetryMetricSchema(BaseModel):
    """Complete set of telemetry thresholds for a module.

    Attributes:
        temperature: Temperature thresholds specific to this module.
        pressure: Pressure thresholds (station-wide standard).
        oxygen: Oxygen concentration thresholds (station-wide standard).
        radiation: Radiation thresholds (station-wide standard).
        humidity: Humidity thresholds (station-wide standard).
        power: Power consumption thresholds (station-wide standard).
    """

    temperature: TemperatureThresholdSchema
    pressure: PressureThresholdSchema
    oxygen: OxygenThresholdSchema
    radiation: RadiationThresholdSchema
    humidity: HumidityThresholdSchema
    power: PowerThresholdSchema


class ModuleSchema(BaseModel):
    """Schema for a single Kuntur Station module.

    Attributes:
        name: Display name of the module.
        function: Primary function description.
        role: Detailed operational role description.
        crew_capacity: Maximum crew this module can hold.
        primary_crew: List of crew role identifiers assigned to this module.
        systems: List of controllable systems in this module.
        telemetry: Alert thresholds for all monitored metrics.
        workstations: Number of scientific workstations (Quetzal only).
        redundancy_class: Redundancy classification (Jaguar only).
        has_docking_port: Whether the module has an external docking port.
        cabins: Number of individual crew cabins (Tucán only).
    """

    name: str
    function: str
    role: str
    crew_capacity: int
    primary_crew: list[str]
    systems: list[str]
    telemetry: TelemetryMetricSchema
    workstations: Optional[int] = None
    redundancy_class: Optional[str] = None
    has_docking_port: Optional[bool] = None
    cabins: Optional[int] = None
