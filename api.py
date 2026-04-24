from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import main


class PatientPayload(BaseModel):
    patient_data: dict[str, Any]


class PatientResultPayload(BaseModel):
    patient_result: dict[str, Any]


class RetrievalOptions(BaseModel):
    patient_data: dict[str, Any]
    candidate_limit: int | None = Field(default=None, ge=1)


app = FastAPI(
    title="Clinical Risk Factor API",
    version="1.0.0",
    description=(
        "Thin API layer over the retrieval-constrained clinical risk factor and "
        "clinical group recommendation pipeline."
    ),
)


def build_retrieval_summary() -> dict[str, Any]:
    patients = main.load_patient_data()
    risk_factors = main.load_risk_factors()
    static_data = main.load_static_data()

    saved_results_path = Path(main.RESULTS_FILE)
    if saved_results_path.exists():
        patient_results = main.normalize_saved_results(main.load_json(saved_results_path))
    else:
        patient_results = []

    risk_candidate_sizes = []
    risk_candidate_counts = []
    static_candidate_sizes = []
    static_candidate_counts = []

    for patient in patients:
        risk_candidates = main.select_risk_factor_candidates(patient)
        risk_candidate_sizes.append(main.estimate_json_size(risk_candidates))
        risk_candidate_counts.append(len(risk_candidates))

    for patient_result in patient_results:
        static_candidates = main.select_static_data_candidates(patient_result, static_data)
        static_candidate_sizes.append(main.estimate_json_size(static_candidates))
        static_candidate_counts.append(len(static_candidates))

    risk_total_size = main.estimate_json_size(risk_factors)
    static_total_size = main.estimate_json_size(static_data)
    avg_risk_candidate_size = mean(risk_candidate_sizes) if risk_candidate_sizes else 0
    avg_static_candidate_size = mean(static_candidate_sizes) if static_candidate_sizes else 0

    risk_reduction = 0.0 if not risk_total_size else 100 * (1 - (avg_risk_candidate_size / risk_total_size))
    static_reduction = 0.0 if not static_total_size else 100 * (1 - (avg_static_candidate_size / static_total_size))

    return {
        "patients": len(patients),
        "risk_factor_reference_count": len(risk_factors),
        "clinical_group_reference_count": len(static_data),
        "risk_factor_candidate_limit": main.RISK_FACTOR_CANDIDATE_LIMIT,
        "clinical_group_candidate_limit": main.CLINICAL_GROUP_CANDIDATE_LIMIT,
        "risk_factor_prompt": {
            "baseline_chars": risk_total_size,
            "average_candidate_chars": round(avg_risk_candidate_size, 2),
            "average_candidate_count": round(mean(risk_candidate_counts), 2) if risk_candidate_counts else 0,
            "reduction_percent": round(risk_reduction, 2),
        },
        "clinical_group_prompt": {
            "baseline_chars": static_total_size,
            "average_candidate_chars": round(avg_static_candidate_size, 2),
            "average_candidate_count": round(mean(static_candidate_counts), 2) if static_candidate_counts else 0,
            "reduction_percent": round(static_reduction, 2),
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": main.model_name,
        "risk_factor_candidate_limit": main.RISK_FACTOR_CANDIDATE_LIMIT,
        "clinical_group_candidate_limit": main.CLINICAL_GROUP_CANDIDATE_LIMIT,
    }


@app.get("/metrics/retrieval")
def retrieval_metrics() -> dict[str, Any]:
    return build_retrieval_summary()


@app.post("/retrieve/risk-factors")
def retrieve_risk_factors(payload: RetrievalOptions) -> dict[str, Any]:
    limit = payload.candidate_limit or main.RISK_FACTOR_CANDIDATE_LIMIT
    candidates = main.select_risk_factor_candidates(payload.patient_data, limit=limit)
    return {
        "candidate_limit": limit,
        "candidate_count": len(candidates),
        "candidate_chars": main.estimate_json_size(candidates),
        "candidates": candidates,
    }


@app.post("/analyze/risk-factors")
def analyze_risk_factors(payload: PatientPayload) -> dict[str, Any]:
    patient_id = payload.patient_data.get("patient_id", "Unknown")
    try:
        llm = main.get_llm()
        risk_factors = main.load_risk_factors()
        results = main.identify_risk_factors(payload.patient_data, risk_factors, llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "patient_id": patient_id,
        "riskFactors": results,
    }


@app.post("/recommend/clinical-group")
def recommend_clinical_group(payload: PatientResultPayload) -> dict[str, Any]:
    try:
        llm = main.get_llm()
        static_data = main.load_static_data()
        recommendation = main.recommend_clinical_group(payload.patient_result, static_data, llm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return recommendation
