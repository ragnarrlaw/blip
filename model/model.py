from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from db.db import Base


class Course(Base):
    __tablename__ = "course"
    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, nullable=False, index=True)
    course_name = Column(String, nullable=False)
    enrollments = Column(Integer, nullable=False, default=10000)

    content_items = relationship("ContentItem", back_populates="course")
    assignments = relationship("Assignment", back_populates="course")


class ContentItem(Base):
    __tablename__ = "content_item"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("course.id"), nullable=False)
    item_key = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    content_kind = Column(String, nullable=False, default="material")
    last_updated = Column(Integer, nullable=False)
    removed_at = Column(String, nullable=True)

    course = relationship("Course", back_populates="content_items")
    versions = relationship("ContentItemVersion", back_populates="content_item")


class ContentItemVersion(Base):
    __tablename__ = "content_item_versions"
    id = Column(Integer, primary_key=True, index=True)
    content_item_id = Column(Integer, ForeignKey("content_item.id"), nullable=False)
    version = Column(Integer, nullable=False)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=True)
    downloaded_at = Column(String, nullable=False)
    chunk_ids = Column(JSON, nullable=True)
    parsed_path = Column(String, nullable=True)
    parsed_at = Column(String, nullable=True)
    parse_error = Column(Text, nullable=True)
    indexed_at = Column(String, nullable=True)
    evicted_at = Column(String, nullable=True)

    content_item = relationship("ContentItem", back_populates="versions")


class Assignment(Base):
    __tablename__ = "assignment"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("course.id"), nullable=False)
    name = Column(String, nullable=False, default="")
    item_key = Column(String, nullable=False, index=True)
    grading_status = Column(String, nullable=False, default="pending")

    course = relationship("Course", back_populates="assignments")
    grading_runs = relationship("GradingRun", back_populates="assignment")
    grades = relationship("Grade", back_populates="assignment")


class AssignmentStatus(Base):
    __tablename__ = "assignment_status"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    grading_status = Column(String, nullable=False, default="pending")
    results_path = Column(String, default="")
    content_type = Column(String, nullable=False)
    content_kind = Column(String, nullable=False, default="short_answer")
    content_hash = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=True)
    last_updated = Column(Integer, nullable=False)
    downloaded_at = Column(String, nullable=False)


class GradingRun(Base):
    __tablename__ = "grading_run"
    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("assignment.id"), nullable=False)
    status = Column(String, nullable=False, default="running")
    params = Column(JSON, nullable=False, default={})
    results_path = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    started_at = Column(String, nullable=False)
    finished_at = Column(String, nullable=True)

    assignment = relationship("Assignment", back_populates="grading_runs")
    content_items = relationship("ContentItem", secondary="grading_run_content")


class GradingRunContent(Base):
    __tablename__ = "grading_run_content"
    run_id = Column(Integer, ForeignKey("grading_run.id", ondelete="CASCADE"), primary_key=True)
    content_item_id = Column(Integer, ForeignKey("content_item.id"), primary_key=True)


class Grade(Base):
    __tablename__ = "grade"
    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("assignment.id"), nullable=False)
    student_id = Column(String, nullable=False)
    question_id = Column(String, nullable=False)
    question_type = Column(String, nullable=False)  # 'independent' or 'scenario'
    question_text = Column(Text, nullable=False)
    scenario_text = Column(Text, nullable=False)
    rubric = Column(Text, nullable=False)
    model_answer = Column(Text, nullable=False)
    student_answer = Column(Text, nullable=False, default="-")
    is_answered = Column(Boolean, nullable=False)
    max_mark = Column(Float, nullable=False)
    mark_assigned = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    params_used = Column(JSON, nullable=False, default={})
    justifications = Column(JSON, nullable=False, default={})
    criterion_scores = Column(JSON, nullable=False, default={})
    assignment = relationship("Assignment", back_populates="grades")


class MasterQuestion(Base):
    __tablename__ = "master_question"

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer, ForeignKey("assignment.id"), nullable=False)
    canonical_id = Column(String, nullable=False)
    question_text = Column(String, nullable=False)
    question_type = Column(String, default="independent")
    scenario_text = Column(String, nullable=True)
    is_scenario_based = Column(Boolean, default=False)
    max_mark = Column(Float, nullable=False)
    model_answer = Column(String, nullable=False)
    rubric = Column(String, nullable=False)

    # Relationships
    assignment = relationship("Assignment")