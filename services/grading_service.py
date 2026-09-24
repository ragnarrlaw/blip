import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from config.conf import Config
from model.model import Grade
from schema.schema import GradingResult, BatchGradingResult
from context.retriever import ChromaRetriever
from services.llm_provider import GeminiProvider, LLMProvider, LLMResponseError

log = logging.getLogger("blip.grader")

SYSTEM_INSTRUCTION = """\
You are an expert academic examiner and calibrated grading assistant.
Your task is to grade structured student answers against a provided rubric, model answer, and course context.

Core Evaluation Directives:
1. Objectivity: Grade strictly based on demonstrated student understanding against the rubric.
2. Evidence-Based Justification: Reference specific phrases from the student's answer when awarding or deducting marks.
3. Calibration & Confidence: Provide a self-reported confidence score between 0.00 and 1.00.
4. Determinism: If an answer relies on alternative valid reasoning not covered in the model answer, explicitly flag it.
5. Strict Formatting: Output valid, parseable JSON matching the requested schema.\
"""


class GradingService:
    def __init__(self, config: Config, provider: LLMProvider | None = None):
        self.config = config
        self.provider = GeminiProvider(config) if provider is None else provider
        self.retriever = ChromaRetriever(config)

    def _sort_chunks_u_shaped(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if len(chunks) <= 2:
            return chunks
        ordered = []
        left = True
        for chunk in chunks:
            if left:
                ordered.insert(0, chunk)
            else:
                ordered.append(chunk)
            left = not left
        return ordered[::-1]

    def _retrieve_context(
        self, course_slug: str, query: str, top_k: int = 3
    ) -> tuple[str, List[str]]:
        collection_name = self.config.chroma_collection(course_slug)
        chunks = self.retriever.search(
            course_name=course_slug,
            collection_name=collection_name,
            query=query,
            top_k=top_k,
        )
        if not chunks:
            return "", []

        ordered_chunks = self._sort_chunks_u_shaped(chunks)
        chunk_snippets = []
        chunk_ids = []
        for i, c in enumerate(ordered_chunks, 1):
            source = c.get("source", "Course Document")
            heading = c.get("heading", "")
            heading_str = f" > {heading}" if heading else ""
            chunk_snippets.append(
                f"--- Context Excerpt [{i}] ({source}{heading_str}) ---\n{c.get('text', '').strip()}"
            )
            chunk_ids.append(source)

        context_body = "\n\n".join(chunk_snippets)
        return (
            f"[COURSE CONTEXT]\nRetrieved course materials:\n{context_body}\n",
            chunk_ids,
        )

    def grade_single(
        self,
        db: Session,
        grade_id: int,
        course_slug: str,
        use_rag: bool = True,
        model_override: Optional[str] = None,
    ) -> Optional[Grade]:
        grade = db.query(Grade).filter(Grade.id == grade_id).first()
        if not grade:
            return None

        if not grade.is_answered or not grade.student_answer.strip():
            grade.mark_assigned = 0.0
            grade.confidence_score = 1.0
            grade.justifications = {
                "summary": "Student provided no answer or dashed response."
            }
            grade.criterion_scores = []
            db.commit()
            return grade

        context_str, chunk_ids = "", []
        if use_rag:
            search_query = f"{grade.question_text}\n{grade.model_answer}"
            context_str, chunk_ids = self._retrieve_context(course_slug, search_query)

        scenario_section = (
            f"[SCENARIO]\n{grade.scenario_text.strip()}\n\n"
            if grade.scenario_text
            else ""
        )

        prompt = f"""\
[EXAMINATION QUESTION]
Max Mark: {grade.max_mark}
Question Statement:
{grade.question_text.strip()}

{scenario_section}[GRADING RUBRIC]
{grade.rubric.strip()}

[MODEL ANSWER]
{grade.model_answer.strip()}

{context_str}[STUDENT ANSWER]
{grade.student_answer.strip()}

[EVALUATION DIRECTIVE]
Evaluate the [STUDENT ANSWER] against the [GRADING RUBRIC] and [MODEL ANSWER].
Return valid JSON matching the schema.
"""
        exec_result = self.provider.generate_structured_response(
            prompt=prompt,
            response_schema=GradingResult,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
            model_override=model_override,
        )

        parsed: GradingResult = exec_result.parsed
        print("exec_results parsed: ", parsed)
        grade.mark_assigned = parsed.mark
        grade.confidence_score = parsed.confidence
        grade.justifications = {
            "summary": parsed.justification,
            "flags": [f.model_dump() for f in parsed.flags],
        }
        grade.criterion_scores = [c.model_dump() for c in parsed.criterion_scores]
        grade.params_used = {
            "use_rag": use_rag,
            "model_name": model_override or self.provider.default_model,
            "chunks_retrieved": chunk_ids,
            "latency_ms": exec_result.latency_ms,
            "tokens": exec_result.total_tokens,
        }

        db.commit()
        db.refresh(grade)
        print("grade: ", grade)
        return grade

    def grade_batch(
        self,
        db: Session,
        grade_ids: List[int],
        course_slug: str,
        use_rag: bool = True,
        model_override: Optional[str] = None,
    ) -> List[Grade]:
        if not grade_ids:
            return []

        grades = db.query(Grade).filter(Grade.id.in_(grade_ids)).all()
        if not grades:
            return []

        ref_grade = grades[0]
        active_grades = []
        for g in grades:
            if not g.is_answered or not g.student_answer.strip():
                g.mark_assigned = 0.0
                g.confidence_score = 1.0
                g.justifications = {"summary": "Student provided no answer."}
                g.criterion_scores = []
            else:
                active_grades.append(g)

        if not active_grades:
            db.commit()
            return grades

        ephemeral_map: Dict[str, Grade] = {}
        student_answers_payload = []
        for idx, g in enumerate(active_grades, 1):
            tag = f"ANS_{idx:02d}"
            ephemeral_map[tag] = g
            student_answers_payload.append(
                f"--- Student Answer Tag: {tag} ---\n{g.student_answer.strip()}"
            )

        context_str, chunk_ids = "", []
        if use_rag:
            search_query = f"{ref_grade.question_text}\n{ref_grade.model_answer}"
            context_str, chunk_ids = self._retrieve_context(course_slug, search_query)

        scenario_section = (
            f"[SCENARIO]\n{ref_grade.scenario_text.strip()}\n\n"
            if ref_grade.scenario_text
            else ""
        )
        joined_answers = "\n\n".join(student_answers_payload)

        prompt = f"""\
[EXAMINATION QUESTION]
Max Mark: {ref_grade.max_mark}
Question Statement:
{ref_grade.question_text.strip()}

{scenario_section}[GRADING RUBRIC]
{ref_grade.rubric.strip()}

[MODEL ANSWER]
{ref_grade.model_answer.strip()}

{context_str}[STUDENT ANSWERS TO EVALUATE]
{joined_answers}

[EVALUATION DIRECTIVE]
Evaluate every student answer against the [GRADING RUBRIC] and [MODEL ANSWER].
Return a JSON array containing evaluations for all provided tags.
"""
        try:
            exec_result = self.provider.generate_structured_response(
                prompt=prompt,
                response_schema=BatchGradingResult,
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                model_override=model_override,
            )
            batch_result: BatchGradingResult = exec_result.parsed

            for eval_item in batch_result.evaluations:
                target_grade = ephemeral_map.get(eval_item.item_tag)
                if not target_grade:
                    continue

                target_grade.mark_assigned = eval_item.mark
                target_grade.confidence_score = eval_item.confidence
                target_grade.justifications = {
                    "summary": eval_item.justification,
                    "flags": [f.model_dump() for f in eval_item.flags],
                }
                target_grade.criterion_scores = [
                    c.model_dump() for c in eval_item.criterion_scores
                ]
                target_grade.params_used = {
                    "use_rag": use_rag,
                    "model_name": model_override or self.provider.default_model,
                    "chunks_retrieved": chunk_ids,
                    "batch_size": len(active_grades),
                    "tokens_per_batch": exec_result.total_tokens,
                }

            db.commit()

        except LLMResponseError as e:
            if e.category == "truncated_output" and len(active_grades) > 1:
                log.warning(
                    f"Batch exceeded tokens. Splitting batch of {len(active_grades)}."
                )
                mid = len(active_grades) // 2
                first_half = [g.id for g in active_grades[:mid]]
                second_half = [g.id for g in active_grades[mid:]]
                self.grade_batch(db, first_half, course_slug, use_rag, model_override)
                self.grade_batch(db, second_half, course_slug, use_rag, model_override)
            else:
                log.error(f"Batch grading failed: {e.message}")
                raise

        return grades

