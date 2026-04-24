# IP_Project - Main Entry Point

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from config folder
env_path = Path(__file__).parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

# Get environment variables
model_endpoint = os.environ.get("MODEL_ENDPOINT")
model_name = os.environ.get("MODEL_NAME")
project_id = os.environ.get("PROJECT_ID")
openai_api_version = os.environ.get("API_VERSION")
client_id = os.environ.get("CLIENT_ID")
client_secret = os.environ.get("CLIENT_SECRET")

# Data paths
DATA_DIR = Path(__file__).parent / "data"
RISK_FACTOR_PROMPT_FILE = DATA_DIR / "generateRFfromLLM.txt"
RECOMMENDATION_PROMPT_FILE = DATA_DIR / "generateRecfromLLM.txt"
DATASET_FILE = DATA_DIR / "patientData.json"
RISKFACTORS_FILE = DATA_DIR / "riskfactors.json"
STATIC_DATA_FILE = DATA_DIR / "staticData.json"
RESULTS_FILE = DATA_DIR / "results.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"
RISK_FACTOR_CANDIDATE_LIMIT = int(os.environ.get("RISK_FACTOR_CANDIDATE_LIMIT", "40"))
CLINICAL_GROUP_CANDIDATE_LIMIT = int(os.environ.get("CLINICAL_GROUP_CANDIDATE_LIMIT", "8"))

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@lru_cache(maxsize=1)
def get_access_token() -> str:
    """Get OAuth2 access token for Azure OpenAI API."""
    auth_url = "https://api.uhg.com/oauth2/token"
    scope = "https://api.uhg.com/.default"
    grant_type = "client_credentials"

    with httpx.Client() as client:
        body = {
            "grant_type": grant_type,
            "scope": scope,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            resp = client.post(auth_url, headers=headers, data=body, timeout=60)
            resp.raise_for_status()
            access_token = resp.json()["access_token"]
            print("  OAuth2 token obtained successfully")
            return access_token
        except httpx.HTTPStatusError as exc:
            print(f"  OAuth2 token request failed: {exc.response.status_code}")
            print(f"    Response: {exc.response.text[:200]}")
            raise
        except Exception as exc:
            print(f"  OAuth2 error: {exc}")
            raise


@lru_cache(maxsize=1)
def get_llm() -> AzureChatOpenAI:
    """Initialize and return the Azure OpenAI LLM client."""
    return AzureChatOpenAI(
        azure_endpoint=model_endpoint,
        api_version=openai_api_version,
        azure_deployment=model_name,
        azure_ad_token=get_access_token(),
        default_headers={"projectId": project_id},
        temperature=0.1,
        max_tokens=4096,
    )


@lru_cache(maxsize=None)
def load_json(filepath: Path) -> Any:
    with open(filepath, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@lru_cache(maxsize=None)
def load_text(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as file_handle:
        return file_handle.read()


def load_patient_data() -> list:
    return load_json(DATASET_FILE)


def load_risk_factors() -> list:
    return load_json(RISKFACTORS_FILE)


def load_static_data() -> list:
    return load_json(STATIC_DATA_FILE)


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def tokenize_text(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def collect_string_values(node: Any) -> list[str]:
    values: list[str] = []

    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.strip():
                values.append(key)
            values.extend(collect_string_values(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(collect_string_values(item))
    elif isinstance(node, str) and node.strip():
        values.append(node)

    return values


def build_text_index(entries: list[dict[str, Any]]) -> dict[str, Any]:
    document_frequency: Counter[str] = Counter()

    for entry in entries:
        document_frequency.update(entry["unique_tokens"])

    total_documents = max(len(entries), 1)
    inverse_document_frequency = {
        token: math.log((1 + total_documents) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }

    return {
        "entries": entries,
        "idf": inverse_document_frequency,
        "size": total_documents,
    }


def score_index_entry(
    query_counter: Counter[str],
    normalized_query_text: str,
    entry: dict[str, Any],
    idf_lookup: dict[str, float],
) -> float:
    overlap_tokens = query_counter.keys() & entry["unique_tokens"]
    token_score = sum(query_counter[token] * idf_lookup.get(token, 1.0) for token in overlap_tokens)

    specificity_penalty = math.sqrt(max(len(entry["unique_tokens"]), 1))
    weighted_token_score = token_score / specificity_penalty

    phrase_boost = 0.0
    for label in entry["labels"]:
        if len(label) > 4 and label in normalized_query_text:
            phrase_boost += 3.0

    overlap_boost = len(overlap_tokens) * 0.15
    return weighted_token_score + phrase_boost + overlap_boost


def search_index(query_text: str, index: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    normalized_query_text = normalize_text(query_text)
    query_counter = Counter(tokenize_text(query_text))
    if not query_counter:
        return []

    scored_entries = []
    for entry in index["entries"]:
        score = score_index_entry(query_counter, normalized_query_text, entry, index["idf"])
        if score > 0:
            scored_entries.append((score, entry))

    scored_entries.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored_entries[:limit]]


def build_patient_search_text(patient_data: dict) -> str:
    return "\n".join(collect_string_values(patient_data))


@lru_cache(maxsize=1)
def build_risk_factor_index() -> dict[str, Any]:
    risk_factors = load_risk_factors()
    entries = []

    for risk_factor_entry in risk_factors:
        labels = {
            normalize_text(value)
            for value in collect_string_values(risk_factor_entry)
            if isinstance(value, str) and value.strip()
        }
        raw_text = "\n".join(sorted(labels))
        tokens = set(tokenize_text(raw_text))
        entries.append(
            {
                "payload": risk_factor_entry,
                "labels": labels,
                "unique_tokens": tokens,
            }
        )

    return build_text_index(entries)


def select_risk_factor_candidates(patient_data: dict, limit: int = RISK_FACTOR_CANDIDATE_LIMIT) -> list:
    patient_text = build_patient_search_text(patient_data)
    candidate_entries = search_index(patient_text, build_risk_factor_index(), limit)
    return [entry["payload"] for entry in candidate_entries]


def collect_static_criteria_strings(node: Any, include_keys: bool = True) -> set[str]:
    labels: set[str] = set()

    if isinstance(node, dict):
        for key, value in node.items():
            if include_keys and isinstance(key, str) and key not in {"High Risk Profile", "Low Risk Profile"}:
                normalized_key = normalize_text(key)
                if normalized_key:
                    labels.add(normalized_key)
            labels.update(collect_static_criteria_strings(value))
    elif isinstance(node, list):
        for item in node:
            labels.update(collect_static_criteria_strings(item, include_keys=False))
    elif isinstance(node, str) and node.strip():
        labels.add(normalize_text(node))

    return labels


@lru_cache(maxsize=1)
def build_static_data_index() -> dict[str, Any]:
    static_data = load_static_data()
    entries = []

    for group_entry in static_data:
        labels = collect_static_criteria_strings(group_entry)
        raw_text = "\n".join(sorted(labels))
        tokens = set(tokenize_text(raw_text))
        entries.append(
            {
                "payload": group_entry,
                "labels": labels,
                "unique_tokens": tokens,
            }
        )

    return build_text_index(entries)


def estimate_json_size(payload: Any) -> int:
    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))


def log_retrieval_stats(label: str, total_items: int, selected_items: int, total_size: int, selected_size: int) -> None:
    reduction = 0.0
    if total_size:
        reduction = 100 * (1 - (selected_size / total_size))
    print(
        f"  {label}: selected {selected_items}/{total_items} items "
        f"({selected_size:,}/{total_size:,} chars, {reduction:.1f}% smaller)"
    )


def build_risk_factor_prompt(patient_data: dict, risk_factors: list) -> str:
    template = load_text(RISK_FACTOR_PROMPT_FILE)
    prompt = template.replace("{{PATIENT_DATA_JSON}}", json.dumps(patient_data, indent=2))
    prompt = prompt.replace("{{RISK_FACTORS_DATABASE_JSON}}", json.dumps(risk_factors, indent=2))
    return prompt


def build_recommendation_prompt(patient_result: dict, static_data_subset: list) -> str:
    template = load_text(RECOMMENDATION_PROMPT_FILE)
    prompt = template.replace("{{PATIENT_RESULTS_JSON}}", json.dumps(patient_result, indent=2))
    prompt = prompt.replace("{{STATIC_DATA_JSON}}", json.dumps(static_data_subset, indent=2))
    return prompt


def parse_llm_json(response_content: str) -> Any:
    content = response_content.strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON response: {exc}")
        print(f"Raw response: {response_content[:500]}...")
        raise


def normalize_risk_factor_response(parsed_response: Any, patient_id: str) -> list:
    if isinstance(parsed_response, list):
        if not parsed_response:
            return []
        if len(parsed_response) == 1 and isinstance(parsed_response[0], dict):
            first_item = parsed_response[0]
            if "riskFactors" in first_item:
                return first_item.get("riskFactors", [])
        if all(isinstance(item, dict) and "riskFactor" in item for item in parsed_response):
            return parsed_response

    if isinstance(parsed_response, dict) and "riskFactors" in parsed_response:
        return parsed_response.get("riskFactors", [])

    raise ValueError(f"Unexpected risk factor response format for patient {patient_id}")


def normalize_saved_results(saved_results: Any) -> list:
    if not isinstance(saved_results, list):
        raise ValueError("results.json must contain a JSON array")

    normalized_results = []
    for item in saved_results:
        if not isinstance(item, dict):
            continue

        patient_id = item.get("patient_id", "Unknown")
        normalized_item = dict(item)
        risk_factors = item.get("riskFactors", [])

        if (
            isinstance(risk_factors, list)
            and len(risk_factors) == 1
            and isinstance(risk_factors[0], dict)
            and "riskFactors" in risk_factors[0]
        ):
            normalized_item["riskFactors"] = risk_factors[0].get("riskFactors", [])
        elif isinstance(risk_factors, dict) and "riskFactors" in risk_factors:
            normalized_item["riskFactors"] = risk_factors.get("riskFactors", [])
        elif risk_factors is None:
            normalized_item["riskFactors"] = []

        if not isinstance(normalized_item.get("riskFactors", []), list):
            raise ValueError(f"Invalid riskFactors payload in results.json for patient {patient_id}")

        normalized_results.append(normalized_item)

    return normalized_results


def collect_patient_labels(risk_factors: list) -> set[str]:
    labels: set[str] = set()

    for item in risk_factors:
        if not isinstance(item, dict):
            continue

        parent = item.get("riskFactor", {})
        parent_risk_factor = parent.get("riskFactor", {})
        parent_label = parent_risk_factor.get("label")
        if isinstance(parent_label, str) and parent_label.strip():
            labels.add(normalize_text(parent_label))

        for clinical_fact in parent.get("clinicalFacts", []):
            fact_label = clinical_fact.get("label")
            if isinstance(fact_label, str) and fact_label.strip():
                labels.add(normalize_text(fact_label))

        for sub_risk_factor in item.get("subRiskFactors", []):
            risk_factor_label = sub_risk_factor.get("riskFactor", {}).get("label")
            if isinstance(risk_factor_label, str) and risk_factor_label.strip():
                labels.add(normalize_text(risk_factor_label))

            for clinical_fact in sub_risk_factor.get("clinicalFacts", []):
                fact_label = clinical_fact.get("label")
                if isinstance(fact_label, str) and fact_label.strip():
                    labels.add(normalize_text(fact_label))

    return labels


def select_static_data_candidates(
    patient_result: dict,
    static_data: list,
    limit: int = CLINICAL_GROUP_CANDIDATE_LIMIT,
) -> list:
    patient_labels = collect_patient_labels(patient_result.get("riskFactors", []))
    if not patient_labels:
        return []

    query_text = "\n".join(sorted(patient_labels))
    candidate_entries = search_index(query_text, build_static_data_index(), limit)
    if not candidate_entries:
        return static_data[:limit]
    return [entry["payload"] for entry in candidate_entries]


def identify_risk_factors(patient_data: dict, risk_factors: list, llm: AzureChatOpenAI) -> list:
    patient_id = patient_data.get("patient_id", "Unknown")
    candidate_risk_factors = select_risk_factor_candidates(patient_data)
    selected_risk_factors = candidate_risk_factors or risk_factors[:RISK_FACTOR_CANDIDATE_LIMIT]
    log_retrieval_stats(
        "Risk factor retrieval",
        len(risk_factors),
        len(selected_risk_factors),
        estimate_json_size(risk_factors),
        estimate_json_size(selected_risk_factors),
    )
    prompt = build_risk_factor_prompt(patient_data, selected_risk_factors)
    response = llm.invoke(prompt)
    parsed_response = parse_llm_json(response.content)
    return normalize_risk_factor_response(parsed_response, patient_id)


def recommend_clinical_group(patient_result: dict, static_data: list, llm: AzureChatOpenAI) -> dict:
    patient_id = patient_result.get("patient_id", "Unknown")
    candidate_groups = select_static_data_candidates(patient_result, static_data)

    if candidate_groups:
        log_retrieval_stats(
            "Clinical group retrieval",
            len(static_data),
            len(candidate_groups),
            estimate_json_size(static_data),
            estimate_json_size(candidate_groups),
        )

    if not candidate_groups:
        return {
            "patient_id": patient_id,
            "recommendedClinicalGroup": None,
            "recommendedProfile": None,
            "matchPercentage": 0,
            "highRiskProfile": None,
            "matchedEvidence": [],
            "lowRiskProfile": None,
            "matchedAudience": [],
            "recommendationStatus": "no_match",
            "reason": "No overlapping criteria were found between the patient results and StaticData.json.",
        }

    prompt = build_recommendation_prompt(patient_result, candidate_groups)
    response = llm.invoke(prompt)
    recommendation = parse_llm_json(response.content)

    if not isinstance(recommendation, dict):
        raise ValueError(f"Unexpected recommendation response format for patient {patient_id}")

    recommendation.setdefault("patient_id", patient_id)
    recommendation.setdefault("highRiskProfile", None)
    recommendation.setdefault("matchedEvidence", [])
    recommendation.setdefault("lowRiskProfile", None)
    recommendation.setdefault("matchedAudience", [])

    match_percentage = recommendation.get("matchPercentage", 0)
    if not isinstance(match_percentage, int):
        raise ValueError(f"Invalid matchPercentage returned for patient {patient_id}")

    if match_percentage <= 70:
        recommendation["recommendedClinicalGroup"] = None
        recommendation["recommendedProfile"] = None
        recommendation["matchedEvidence"] = recommendation.get("matchedEvidence", []) or []
        recommendation["matchedAudience"] = recommendation.get("matchedAudience", []) or []
        recommendation["recommendationStatus"] = "no_match"
        recommendation["reason"] = "No supplied clinical group exceeded the 70% threshold."

    return recommendation


def save_json(payload: Any, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=4)
    print(f"Saved: {output_path}")


def generate_risk_factor_results() -> list:
    print("Loading patient and risk factor data...")
    patient_data_list = load_patient_data()
    risk_factors = load_risk_factors()
    build_risk_factor_index()
    print(f"  - Loaded {len(patient_data_list)} patient(s)")
    print(f"  - Loaded {len(risk_factors)} reference risk factors")
    print(f"  - Risk factor candidate limit: {RISK_FACTOR_CANDIDATE_LIMIT}")
    print()

    llm = get_llm()
    all_results = []

    for patient_index, patient in enumerate(patient_data_list, 1):
        patient_id = patient.get("patient_id", "Unknown")
        print(f"\n{'=' * 60}")
        print(f"Processing Patient {patient_index}/{len(patient_data_list)}: {patient_id}")
        print("=" * 60)
        print("Sending risk factor request to LLM...")

        try:
            risk_factor_results = identify_risk_factors(patient, risk_factors, llm)
            patient_result = {
                "patient_id": patient_id,
                "riskFactors": risk_factor_results,
            }
            all_results.append(patient_result)

            print(f"\nIdentified {len(risk_factor_results)} risk factor(s):")
            print("-" * 40)

            for index, risk_factor in enumerate(risk_factor_results, 1):
                rf_label = risk_factor.get("riskFactor", {}).get("riskFactor", {}).get("label", "N/A")
                rf_value = risk_factor.get("riskFactor", {}).get("riskFactor", {}).get("value")
                clinical_facts = risk_factor.get("riskFactor", {}).get("clinicalFacts", [])
                sub_risk_factors = risk_factor.get("subRiskFactors", [])

                status = "MATCHED" if rf_value is not None else "SUGGESTED"
                print(f"\n{index}. [{status}] {rf_label[:80]}...")

                if clinical_facts:
                    print(f"   Clinical Facts ({len(clinical_facts)}):")
                    for clinical_fact in clinical_facts[:3]:
                        print(f"     - {clinical_fact.get('label', 'N/A')}")
                    if len(clinical_facts) > 3:
                        print(f"     ... and {len(clinical_facts) - 3} more")

                if sub_risk_factors:
                    print(f"   Sub-Risk Factors ({len(sub_risk_factors)}):")
                    for sub_risk_factor in sub_risk_factors[:2]:
                        sub_label = sub_risk_factor.get("riskFactor", {}).get("label", "N/A")
                        print(f"     - {sub_label[:60]}...")
                    if len(sub_risk_factors) > 2:
                        print(f"     ... and {len(sub_risk_factors) - 2} more")

            print(f"\nPatient {patient_id} risk factor analysis complete")
        except Exception as exc:
            print(f"\nError analyzing patient {patient_id}: {exc}")
            all_results.append({
                "patient_id": patient_id,
                "error": str(exc),
                "riskFactors": [],
            })

    return all_results


def generate_recommendations(patient_results: list) -> list:
    static_data = load_static_data()
    build_static_data_index()
    print(f"\nLoaded {len(static_data)} clinical group definition set(s) from StaticData.json")
    print(f"  - Clinical group candidate limit: {CLINICAL_GROUP_CANDIDATE_LIMIT}")

    llm = get_llm()
    recommendations = []

    for patient_result in patient_results:
        patient_id = patient_result.get("patient_id", "Unknown")
        print(f"\nEvaluating clinical group recommendation for patient: {patient_id}")

        if patient_result.get("error"):
            recommendation = {
                "patient_id": patient_id,
                "recommendedClinicalGroup": None,
                "recommendedProfile": None,
                "matchPercentage": 0,
                "highRiskProfile": None,
                "matchedEvidence": [],
                "lowRiskProfile": None,
                "matchedAudience": [],
                "recommendationStatus": "skipped",
                "reason": f"Risk factor generation failed: {patient_result['error']}",
            }
            recommendations.append(recommendation)
            print("  Skipped because risk factor generation failed")
            continue

        try:
            recommendation = recommend_clinical_group(patient_result, static_data, llm)
            recommendations.append(recommendation)

            clinical_group = recommendation.get("recommendedClinicalGroup")
            profile = recommendation.get("recommendedProfile")
            match_percentage = recommendation.get("matchPercentage")

            if clinical_group:
                print(
                    f"  Recommended clinical group: {clinical_group} "
                    f"({profile}, {match_percentage}% match)"
                )
            else:
                print("  No clinical group recommendation met the 70% threshold")
        except Exception as exc:
            recommendations.append({
                "patient_id": patient_id,
                "recommendedClinicalGroup": None,
                "recommendedProfile": None,
                "matchPercentage": 0,
                "highRiskProfile": None,
                "matchedEvidence": [],
                "lowRiskProfile": None,
                "matchedAudience": [],
                "recommendationStatus": "error",
                "reason": str(exc),
            })
            print(f"  Recommendation failed: {exc}")

    return recommendations


def main() -> None:
    print("=" * 60)
    print("CLINICAL RISK FACTOR AND GROUP RECOMMENDATION SYSTEM")
    print("=" * 60)
    print(f"Model: {model_name}")
    print()

    patient_results = generate_risk_factor_results()
    save_json(patient_results, RESULTS_FILE)

    recommendations = generate_recommendations(patient_results)
    save_json(recommendations, RECOMMENDATIONS_FILE)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total patients processed: {len(patient_results)}")
    print(f"Recommendations generated: {len(recommendations)}")
    print(f"Results file: {RESULTS_FILE}")
    print(f"Recommendations file: {RECOMMENDATIONS_FILE}")


if __name__ == "__main__":
    main()
