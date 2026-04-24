import argparse
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main


def build_retrieval_report() -> dict:
    patients = main.load_patient_data()
    risk_factors = main.load_risk_factors()
    static_data = main.load_static_data()

    risk_candidate_sizes = []
    risk_candidate_counts = []
    risk_candidate_labels = []

    for patient in patients:
        candidates = main.select_risk_factor_candidates(patient)
        risk_candidate_sizes.append(main.estimate_json_size(candidates))
        risk_candidate_counts.append(len(candidates))
        risk_candidate_labels.append(
            [
                item.get("riskFactor", {}).get("riskFactor", {}).get("label", "N/A")
                for item in candidates[:5]
            ]
        )

    patient_results = []
    if Path(main.RESULTS_FILE).exists():
        patient_results = main.normalize_saved_results(main.load_json(main.RESULTS_FILE))

    static_candidate_sizes = []
    static_candidate_counts = []
    static_candidate_names = []
    for patient_result in patient_results:
        candidates = main.select_static_data_candidates(patient_result, static_data)
        static_candidate_sizes.append(main.estimate_json_size(candidates))
        static_candidate_counts.append(len(candidates))

        names = []
        for group in candidates[:5]:
            if isinstance(group, dict) and group:
                names.append(next(iter(group.keys())))
        static_candidate_names.append(names)

    risk_total_size = main.estimate_json_size(risk_factors)
    static_total_size = main.estimate_json_size(static_data)
    avg_risk_candidate_size = mean(risk_candidate_sizes) if risk_candidate_sizes else 0
    avg_static_candidate_size = mean(static_candidate_sizes) if static_candidate_sizes else 0

    return {
        "dataset": {
            "patients": len(patients),
            "risk_factor_reference_count": len(risk_factors),
            "clinical_group_reference_count": len(static_data),
            "saved_patient_results": len(patient_results),
        },
        "risk_factor_retrieval": {
            "candidate_limit": main.RISK_FACTOR_CANDIDATE_LIMIT,
            "baseline_chars": risk_total_size,
            "average_candidate_chars": round(avg_risk_candidate_size, 2),
            "average_candidate_count": round(mean(risk_candidate_counts), 2) if risk_candidate_counts else 0,
            "reduction_percent": round(100 * (1 - (avg_risk_candidate_size / risk_total_size)), 2) if risk_total_size else 0,
            "top_labels_per_patient": risk_candidate_labels,
        },
        "clinical_group_retrieval": {
            "candidate_limit": main.CLINICAL_GROUP_CANDIDATE_LIMIT,
            "baseline_chars": static_total_size,
            "average_candidate_chars": round(avg_static_candidate_size, 2),
            "average_candidate_count": round(mean(static_candidate_counts), 2) if static_candidate_counts else 0,
            "reduction_percent": round(100 * (1 - (avg_static_candidate_size / static_total_size)), 2) if static_total_size and static_candidate_sizes else 0,
            "top_groups_per_patient_result": static_candidate_names,
        },
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval efficiency for the project sample dataset.")
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON report.")
    args = parser.parse_args()

    report = build_retrieval_report()
    print(json.dumps(report, indent=2))

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main_cli()
