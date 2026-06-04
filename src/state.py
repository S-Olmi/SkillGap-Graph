from typing import TypedDict, List, Dict, Optional, Any, Literal, Annotated
import operator


SkillStatus = Literal["direct_match", "semantic_match", "partial", "missing", "unknown"]
DataSource = Literal["cv", "github", "rag", "job", "publications"]

class Evidence(TypedDict):
    source: DataSource
    content: str

class DecisionStep(TypedDict):
    step: str
    rule: str
    inputs_used: List[str]
    output: Any

class CanonicalSkill(TypedDict):
    original: str
    capability: str
    dimensions: List[str]
    source: DataSource

class ExplanatoryContext(TypedDict):
    status: SkillStatus
    snippets: List[Evidence]
    confidence: float
    dimensions: List[str]

class AgentState(TypedDict):
    cv_raw_text: str
    job_raw_text: Optional[str]
    available_sources: Dict[str, Dict[str, Any]]

    candidate_skills: Annotated[List[Dict[str, Any]], operator.add]
    role_requirements: Annotated[List[Dict[str, Any]], operator.add]
    soft_requirements: Annotated[List[Dict[str, Any]], operator.add]

    skill_gap: Dict[str, Dict[str, str]]
    match_score: float
    canonical_skills: Annotated[List[CanonicalSkill], operator.add]
    potential_rag_targets: Dict[str, Dict[str, str]]

    triage_decision: Literal["ACCEPT", "REJECT"]
    analysis_path: Literal["LIGHT", "DEEP"]

    rag_ready: bool
    explanatory_context: Dict[str, ExplanatoryContext]

    report: Optional[str]
    decisions: Annotated[List[DecisionStep], operator.add]