import pandas as pd
import io
from sqlalchemy.orm import Session
from model.model import Grade, Assignment


def generate_assignment_report(db: Session, assignment_id: int) -> io.BytesIO:
    assignment = db.query(Assignment).filter(Assignment.id == assignment_id).first()
    grades = db.query(Grade).filter(Grade.assignment_id == assignment_id).all()

    if not grades:
        raise ValueError("No grades found for this assignment.")

    raw_data = []
    criteria_data = []

    for g in grades:
        summary = g.justifications.get("summary", "") if g.justifications else ""
        flags = g.justifications.get("flags", []) if g.justifications else []
        flag_notes = "; ".join([f"{flg.get('flag_type')}: {flg.get('reason')}" for flg in flags])
        params = g.params_used or {}

        # Safely parse the criterion scores depending on whether it saved as a list or dict
        c_scores = g.criterion_scores or []
        crit_summary_parts = []

        if isinstance(c_scores, list):
            for idx, c in enumerate(c_scores):
                crit_name = c.get("criterion", f"Criterion {idx + 1}")
                awarded = c.get("allocated_mark", 0)
                c_max = c.get("max_mark", 0)
                reason = c.get("reason", "")

                crit_summary_parts.append(f"{crit_name}: {awarded}/{c_max}")

                criteria_data.append({
                    "Student ID": str(g.student_id),
                    "Question ID": str(g.question_id),
                    "Criterion": str(crit_name),
                    "Awarded Mark": float(awarded),
                    "Max Mark": float(c_max),
                    "Reasoning": str(reason)
                })
        elif isinstance(c_scores, dict):
            for k, v in c_scores.items():
                crit_summary_parts.append(f"{k}: {v}")
                criteria_data.append({
                    "Student ID": str(g.student_id),
                    "Question ID": str(g.question_id),
                    "Criterion": str(k),
                    "Awarded Mark": float(v),
                    "Max Mark": "-",
                    "Reasoning": "-"
                })

        raw_data.append({
            "Student ID": str(g.student_id),
            "Question ID": str(g.question_id),
            "Max Mark": float(g.max_mark),
            "Awarded Mark": float(g.mark_assigned),
            "Criteria Summary": " | ".join(crit_summary_parts),
            "AI Confidence": float(g.confidence_score),
            "Justification": summary,
            "Flags/Alerts": flag_notes,
            "RAG Enabled": params.get("use_rag", False),
            "Model Used": params.get("model_name", "Unknown"),
            "Tokens Used": params.get("tokens_per_batch", 0),
        })

    df = pd.DataFrame(raw_data)
    df_criteria = pd.DataFrame(criteria_data) if criteria_data else pd.DataFrame(
        columns=["Student ID", "Question ID", "Criterion", "Awarded Mark", "Max Mark", "Reasoning"])

    # Sheet 1: Lecturer Gradebook (Dynamic Matrix)
    df_matrix = df.pivot_table(
        index="Student ID",
        columns="Question ID",
        values="Awarded Mark",
        aggfunc="first"
    )
    df_matrix["Total Awarded"] = df.groupby("Student ID")["Awarded Mark"].sum()
    df_matrix["Total Possible"] = df.groupby("Student ID")["Max Mark"].sum()

    question_cols = [c for c in df_matrix.columns if c not in ["Total Awarded", "Total Possible"]]
    cols = ["Total Awarded", "Total Possible"] + sorted(question_cols)
    df_matrix = df_matrix[cols].reset_index()

    # Sheet 2: Audit Trail
    df_audit = df[[
        "Student ID", "Question ID", "Max Mark", "Awarded Mark",
        "Criteria Summary", "AI Confidence", "Flags/Alerts", "Justification"
    ]]

    # Sheet 3: Research Telemetry
    df_telemetry = df[[
        "Student ID", "Question ID", "RAG Enabled", "Model Used",
        "Tokens Used", "AI Confidence"
    ]].rename(columns={"AI Confidence": "Raw Confidence"})

    # Force all column names to strings to prevent XlsxWriter float crashes
    df_matrix.columns = [str(c) for c in df_matrix.columns]
    df_audit.columns = [str(c) for c in df_audit.columns]
    df_telemetry.columns = [str(c) for c in df_telemetry.columns]
    df_criteria.columns = [str(c) for c in df_criteria.columns]

    # Write to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_matrix.to_excel(writer, sheet_name='Gradebook Matrix', index=False)
        df_audit.to_excel(writer, sheet_name='Moderation Audit', index=False)
        df_criteria.to_excel(writer, sheet_name='Criteria Breakdown', index=False)
        df_telemetry.to_excel(writer, sheet_name='Research Telemetry', index=False)

        # Auto-adjust column widths safely
        for sheet_name, df_sheet in zip(
                ['Gradebook Matrix', 'Moderation Audit', 'Criteria Breakdown', 'Research Telemetry'],
                [df_matrix, df_audit, df_criteria, df_telemetry]
        ):
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df_sheet.columns):
                col_data_len = df_sheet[col].astype(str).str.len().max()
                if pd.isna(col_data_len):
                    col_data_len = 0

                max_len = int(max(col_data_len, len(str(col)))) + 2
                worksheet.set_column(i, i, min(max_len, 60))

    output.seek(0)
    return output, assignment.name
