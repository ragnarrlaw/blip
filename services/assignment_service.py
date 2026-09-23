from sqlalchemy.orm import Session
from model.model import Assignment


def register_moodle_assignment(
        db: Session,
        course_id: int,
        cmid: int,
        assignment_name: str
) -> int:
    """
    Ensures the Moodle assignment exists in the database.
    Returns the internal database ID of the assignment.
    """
    item_key = f"quiz:{cmid}"

    assignment = db.query(Assignment).filter(Assignment.item_key == item_key).first()

    if not assignment:
        assignment = Assignment(
            course_id=course_id,
            name=assignment_name,
            item_key=item_key,
            grading_status="pending"
        )
        db.add(assignment)
        db.commit()
        db.refresh(assignment)

    return assignment.id