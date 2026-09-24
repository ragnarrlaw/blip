from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GradeBase(BaseModel):
    student_id: str
    question_id: str
    question_type: str = Field(pattern="^(independent|scenario)$")
    question_text: str
    scenario_text: str
    rubric: str
    model_answer: str
    student_answer: str = "-"
    is_answered: bool
    max_mark: float
    mark_assigned: float
    confidence_score: float
    params_used: Dict[str, Any] = {}
    justifications: Dict[str, Any] = {}
    criterion_scores: Dict[str, float] = {}


class GradeCreate(GradeBase):
    assignment_id: int


class GradeResponse(GradeBase):
    id: int
    assignment_id: int

    class Config:
        from_attributes = True


class AssignmentBase(BaseModel):
    name: str
    item_key: str
    grading_status: str = "pending"


class AssignmentResponse(AssignmentBase):
    id: int
    course_id: int

    class Config:
        from_attributes = True


class ExtractedMasterQuestion(BaseModel):
    canonical_id: str = Field(
        description="The question identifier, e.g., 'Q1' or 'Q3_a'."
    )
    question_type: str = Field(
        pattern="^(independent|scenario)$",
        description="Must be 'independent' or 'scenario'.",
    )
    question_text: str = Field(description="The exact wording of the question.")
    scenario_text: Optional[str] = Field(
        None,
        description="The shared background text if this is a scenario-based question. Null for independent questions.",
    )
    max_mark: float = Field(description="The maximum marks allocated to this question.")
    model_answer: str = Field(
        description="A comprehensive, ideal answer to the question."
    )
    rubric: str = Field(
        description="A detailed, point-by-point breakdown of how to award marks. Use Markdown bullet points."
    )


class MasterQuestionList(BaseModel):
    questions: List[ExtractedMasterQuestion]


class ParsedAnswer(BaseModel):
    canonical_id: str = Field(
        description="The canonical ID of the master question (e.g., 'Q6_1')."
    )
    student_answer: str = Field(
        description="The isolated text the student wrote for this specific question."
    )
    is_blank: bool = Field(
        description="True if the student completely skipped or left this sub-question blank."
    )


class ParsedStudentSubmission(BaseModel):
    ephemeral_id: str = Field(
        description="The temporary ID assigned to this raw submission (e.g., 'RAW_01')."
    )
    answers: List[ParsedAnswer]


class BatchParsedSubmissions(BaseModel):
    submissions: List[ParsedStudentSubmission]


class CriterionScore(BaseModel):
    criterion: str = Field(description="Name or ID of the rubric criterion.")
    allocated_mark: float = Field(description="Mark awarded for this criterion.")
    max_mark: float = Field(description="Maximum mark possible for this criterion.")
    reason: str = Field(description="Brief justification for this specific score.")


class GradingFlag(BaseModel):
    type: str = Field(
        description="Flag type: 'low_confidence', 'alternative_solution', 'unclear_response', or 'grading_discrepancy'."
    )
    detail: str = Field(description="Explanation of the flag.")


class GradingResult(BaseModel):
    mark: float = Field(description="Numeric mark awarded.")
    criterion_scores: List[CriterionScore] = Field(
        description="Itemized criteria breakdown."
    )
    justification: str = Field(description="Comprehensive examiner summary.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.00 and 1.00."
    )
    flags: List[GradingFlag] = Field(
        default_factory=list, description="Any evaluation flags."
    )


class BatchStudentEvaluation(GradingResult):
    item_tag: str = Field(
        description="The ephemeral tag matching the student answer (e.g., 'ANS_01')."
    )


class BatchGradingResult(BaseModel):
    evaluations: List[BatchStudentEvaluation]


class GradingExecutionFilter(BaseModel):
    student_ids: Optional[List[str]] = Field(
        None, description="List of specific student IDs to grade."
    )
    question_ids: Optional[List[str]] = Field(
        None, description="List of specific canonical IDs, e.g., ['Q1', 'Q2']."
    )
    question_types: Optional[List[str]] = Field(
        None, description="e.g., ['scenario', 'independent']."
    )
    force_regrade: bool = Field(
        False, description="If True, overwrites previously graded records."
    )
    use_rag: bool = Field(
        True, description="Enable or disable ChromaDB context retrieval."
    )
    model_name: Optional[str] = Field(
        None, description="Override model, e.g., 'gemini-1.5-pro'."
    )
    batch_size: int = Field(15, description="Number of answers per API batch call.")

