from pydantic import BaseModel


class RagQuestionSeed(BaseModel):
    """A seed for a RAG-dependent question where at least one tool parameter
    requires consulting MASA-SEC protocol documents.

    The sensor_reading appears in the generated query. The student must compare
    it against thresholds in technical documents to determine the correct
    severity (send_alert) or protocol_id (activate_protocol).

    Attributes:
        seed_id: Unique identifier for this seed.
        tool_name: Either 'send_alert' or 'activate_protocol'.
        tool_call: Pre-built canonical tool call string with all parameters.
        module: Station module where the incident occurs.
        metric: Type of sensor reading (pressure, oxygen, radiation, etc.).
        sensor_reading: Exact value to embed in the query text.
        incident_description: Brief scenario description for LLM context.
        rag_requirement: Explanation of what the student must look up in docs.
        doc_id: Primary document ID containing the threshold information.
        protocol_id: MASA-SEC protocol governing this threshold.
        phrasing_index: Variant index for phrasing diversity.
    """

    seed_id: str
    tool_name: str
    tool_call: str
    module: str
    metric: str
    sensor_reading: str
    incident_description: str
    rag_requirement: str
    doc_id: str
    protocol_id: str
    phrasing_index: int = 0
