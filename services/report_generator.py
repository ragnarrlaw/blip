import pandas as pd
import io
import re
from sqlalchemy.orm import Session
from model.model import Grade, Assignment, MasterQuestion


def natural_keys(text):
    """Sorts strings containing numbers naturally (Q2 before Q10)."""
    return tuple(int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text)))


def format_q_id(canonical_id: str, scenario_text: str) -> str:
    """Prefixes scenario-based questions with SQ- for visual clarity and sorting."""
    cid = str(canonical_id)
    if scenario_text and scenario_text != "-":
        return cid.replace("Q", "SQ-") if cid.upper().startswith("Q") else f"SQ-{cid}"
    return cid


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

        # Apply the new visual formatting
        display_id = format_q_id(g.question_id, g.scenario_text)

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
                    "Question ID": display_id,
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
                    "Question ID": display_id,
                    "Criterion": str(k),
                    "Awarded Mark": float(v),
                    "Max Mark": "-",
                    "Reasoning": "-"
                })

        raw_data.append({
            "Student ID": str(g.student_id),
            "Question ID": display_id,
            "Scenario Text": str(g.scenario_text) if g.scenario_text else "-",
            "Question Text": str(g.question_text),
            "Student Answer": str(g.student_answer),
            "Max Mark": float(g.max_mark),
            "Awarded Mark": float(g.mark_assigned),
            "Criteria Summary": " | ".join(crit_summary_parts),
            "Justification": summary,
            "AI Confidence": float(g.confidence_score),
            "Flags/Alerts": flag_notes,
            "RAG Enabled": params.get("use_rag", False),
            "Model Used": params.get("model_name", "Unknown"),
            "Tokens Used": params.get("tokens_per_batch", 0),
        })

    df = pd.DataFrame(raw_data)

    # Sort data naturally by Question ID so the Dossier flows sequentially (Q's then SQ's)
    df['sort_key'] = df['Question ID'].apply(natural_keys)
    df = df.sort_values(by=['Student ID', 'sort_key']).drop('sort_key', axis=1)

    # 1. Sheet 1: Final Gradebook
    df_gradebook = df.groupby("Student ID").agg(
        Total_Awarded=("Awarded Mark", "sum")
    ).reset_index()

    assignment_total_possible = df.groupby("Student ID")["Max Mark"].sum().max()
    df_gradebook["Total Possible"] = assignment_total_possible
    df_gradebook.rename(columns={"Total_Awarded": "Total Awarded"}, inplace=True)

    # 2. Sheet 2: Moderation Dossier
    df_dossier = df[[
        "Student ID", "Question ID", "Scenario Text", "Question Text",
        "Student Answer", "Max Mark", "Awarded Mark",
        "Criteria Summary", "Justification", "AI Confidence", "Flags/Alerts"
    ]]

    # 3. Sheet 3: Criteria Breakdown
    df_criteria = pd.DataFrame(criteria_data) if criteria_data else pd.DataFrame(
        columns=["Student ID", "Question ID", "Criterion", "Awarded Mark", "Max Mark", "Reasoning"])

    # 4. Sheet 4: Research Telemetry
    df_telemetry = df[[
        "Student ID", "Question ID", "RAG Enabled", "Model Used",
        "Tokens Used", "AI Confidence"
    ]].rename(columns={"AI Confidence": "Raw Confidence"})

    # 5. Sheet 5: Question Key
    master_qs = db.query(MasterQuestion).filter(MasterQuestion.assignment_id == assignment_id).all()
    key_data = [{
        "Question ID": format_q_id(mq.canonical_id, mq.scenario_text),
        "Max Mark": float(mq.max_mark),
        "Question Type": str(mq.question_type),
        "Question Text": str(mq.question_text),
        "Scenario Text": str(mq.scenario_text) if mq.scenario_text else "-"
    } for mq in master_qs]

    df_key = pd.DataFrame(key_data)
    if not df_key.empty:
        df_key = df_key.sort_values(by="Question ID", key=lambda x: x.map(natural_keys))

    # Force all column names to strings to prevent XlsxWriter float crashes
    df_gradebook.columns = [str(c) for c in df_gradebook.columns]
    df_dossier.columns = [str(c) for c in df_dossier.columns]
    df_criteria.columns = [str(c) for c in df_criteria.columns]
    df_telemetry.columns = [str(c) for c in df_telemetry.columns]
    if not df_key.empty:
        df_key.columns = [str(c) for c in df_key.columns]

    # Write to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_gradebook.to_excel(writer, sheet_name='Final Gradebook', index=False)
        df_dossier.to_excel(writer, sheet_name='Moderation Dossier', index=False)
        df_criteria.to_excel(writer, sheet_name='Criteria Breakdown', index=False)
        df_telemetry.to_excel(writer, sheet_name='Research Telemetry', index=False)
        if not df_key.empty:
            df_key.to_excel(writer, sheet_name='Question Key', index=False)

        # Auto-adjust column widths safely
        sheets_to_format = [
            ('Final Gradebook', df_gradebook),
            ('Moderation Dossier', df_dossier),
            ('Criteria Breakdown', df_criteria),
            ('Research Telemetry', df_telemetry),
            ('Question Key', df_key)
        ]

        for sheet_name, df_sheet in sheets_to_format:
            if df_sheet.empty and sheet_name == 'Question Key':
                continue

            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df_sheet.columns):
                col_data_len = df_sheet[col].astype(str).str.len().max()
                if pd.isna(col_data_len):
                    col_data_len = 0

                max_len = int(max(col_data_len, len(str(col)))) + 2

                if col in ["Scenario Text", "Question Text", "Student Answer", "Justification", "Criteria Summary",
                           "Reasoning"]:
                    worksheet.set_column(i, i, 60)
                else:
                    worksheet.set_column(i, i, min(max_len, 40))

    output.seek(0)
    return output, assignment.name