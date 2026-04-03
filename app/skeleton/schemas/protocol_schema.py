from typing import Any, Optional
from pydantic import BaseModel
from enum import Enum


class SeverityLevel(str, Enum):
    """Valid severity levels for MASA security protocols and alerts.

    These map directly to the send_alert tool's severity parameter.
    """

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ProtocolScope(str, Enum):
    """Valid activation scopes for MASA security protocols.

    These map directly to the activate_protocol tool's scope parameter.
    """

    module_only = "module_only"
    station_wide = "station_wide"


class TriggerThresholdSchema(BaseModel):
    """Precise numeric trigger conditions for a security protocol.

    Attributes:
        metric: Name of the monitored metric that triggers this protocol.
        operator: Comparison operator (e.g., less_than, greater_than, between).
        value: Primary trigger value. None for event-based triggers.
        unit: Unit of the trigger value. None for event-based triggers.
        value_min: Lower bound for between-operator triggers.
        value_max: Upper bound for between-operator triggers.
        value_altitude_km: Altitude deviation threshold in km (MASA-SEC-016).
        value_velocity_ms: Velocity delta threshold in m/s (MASA-SEC-016).
        sustained_minutes: Minimum sustained duration before trigger activates.
    """

    metric: str
    operator: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    value_altitude_km: Optional[float] = None
    value_velocity_ms: Optional[float] = None
    sustained_minutes: Optional[int] = None


class SecurityProtocolSchema(BaseModel):
    """Schema for a single MASA security protocol (MASA-SEC-XXX).

    Attributes:
        name: Human-readable protocol name.
        description: Plain-language description of when this protocol applies.
        trigger_condition: Plain-language trigger description for documentation.
        trigger_thresholds: Precise numeric conditions for programmatic use.
        severity: Alert severity level when this protocol is activated.
        scope: Whether activation affects only the local module or the full station.
        required_actions: Ordered list of actions to execute upon activation.
    """

    name: str
    description: str
    trigger_condition: str
    trigger_thresholds: TriggerThresholdSchema
    severity: SeverityLevel
    scope: ProtocolScope
    required_actions: list[str]
