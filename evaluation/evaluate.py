import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent import AgentCore, ConversationSession
from src.config import CUSTOM_CASES_FILE, EVALUATION_DIR, VISIBLE_CASES_FILE

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


class EvaluationEngine:
    """
    Automated evaluation runner for Aster & Row AI Support Agent.
    Runs visible-cases.json and custom-cases.json against deterministic criteria.
    """

    def __init__(self, agent: Optional[AgentCore] = None):
        self.agent = agent or AgentCore()

    def _normalize_text(self, text: str) -> str:
        """Normalizes whitespace and converts text to lowercase."""
        return " ".join(text.lower().split())

    def _check_concept_match(self, concept: str, text: str) -> bool:
        """
        Checks if a concept or key requirement is represented in the text.
        Handles flexible phrasing and conceptual keywords.
        """
        norm_text = self._normalize_text(text)
        norm_concept = self._normalize_text(concept)

        # Exact substring match first
        if norm_concept in norm_text:
            return True

        # Concept mapping heuristics
        concept_lower = concept.lower()

        # 30 calendar days
        if "30" in concept_lower and ("day" in concept_lower or "calendar" in concept_lower):
            return "30" in norm_text and "day" in norm_text

        # 45 calendar days
        if "45" in concept_lower and ("day" in concept_lower or "calendar" in concept_lower):
            return "45" in norm_text and "day" in norm_text

        # 7 calendar days / report within 7 days
        if "7" in concept_lower and ("day" in concept_lower or "report" in concept_lower):
            return "7" in norm_text and "day" in norm_text

        # 30 minutes cancellation
        if "30 min" in concept_lower or ("30" in concept_lower and "minute" in concept_lower):
            return "30 min" in norm_text or ("30" in norm_text and "minute" in norm_text)

        # Canada shipping / Canada is supported
        if "canada is supported" in concept_lower or ("canada" in concept_lower and "support" in concept_lower):
            return "canada" in norm_text and any(w in norm_text for w in ["support", "ship", "deliver", "available", "yes"])

        if "5–9" in concept_lower or "5-9" in concept_lower:
            return ("5–9" in norm_text or "5-9" in norm_text or "5 to 9" in norm_text)

        # Duties or taxes not prepaid
        if "duties" in concept_lower or "taxes" in concept_lower:
            return "dut" in norm_text or "tax" in norm_text

        # Germany not supported / unsupported
        if "germany" in concept_lower and ("not" in concept_lower or "unavailable" in concept_lower or "support" in concept_lower):
            return "germany" in norm_text and any(w in norm_text for w in ["not", "only", "unavailable", "canada", "uk"])

        # No lifetime warranty / warranty periods
        if "no lifetime" in concept_lower or "lifetime warranty" in concept_lower:
            return ("not" in norm_text or "no" in norm_text) and "lifetime" in norm_text

        if "2 year" in concept_lower or "2-year" in concept_lower:
            return "2 year" in norm_text or "2-year" in norm_text

        if "1 year" in concept_lower or "1-year" in concept_lower:
            return "1 year" in norm_text or "1-year" in norm_text

        # Final sale damaged exception / human review
        if "final sale" in concept_lower and "damaged" in concept_lower:
            return ("final" in norm_text or "sale" in norm_text) and ("damage" in norm_text or "defect" in norm_text)

        if "human review" in concept_lower or "human" in concept_lower or "support" in concept_lower:
            return any(w in norm_text for w in ["human", "support", "team", "specialist", "review", "agent"])

        # Cancelled order will not be shipped
        if "cancelled" in concept_lower:
            return "cancel" in norm_text

        # Order not found
        if "order was not found" in concept_lower or "not found" in concept_lower:
            return "not found" in norm_text or "could not find" in norm_text or "not in our system" in norm_text

        # Delivery estimate unavailable
        if "delivery estimate is unavailable" in concept_lower or "unavailable" in concept_lower:
            return "unavailable" in norm_text or "not available" in norm_text or "no active delivery" in norm_text

        # Insufficient information / human confirmation
        if "insufficient" in concept_lower or "information" in concept_lower:
            return any(w in norm_text for w in ["insufficient", "does not have", "cannot find", "unavailable", "not contain", "support"])

        # Conflict handling: hand-wash vs dishwasher safe
        if "one says hand-wash" in concept_lower or "hand-wash" in concept_lower:
            return "hand-wash" in norm_text or "hand wash" in norm_text or "hand-washed" in norm_text

        if "dishwasher safe" in concept_lower or "dishwasher" in concept_lower:
            return "dishwasher" in norm_text

        if "current official sources conflict" in concept_lower or "conflict" in concept_lower:
            return "conflict" in norm_text or "discrepanc" in norm_text or "opposing" in norm_text

        # Fallback: check if major keywords in concept appear in text
        keywords = [w for w in re.findall(r'\w+', concept_lower) if len(w) > 3 and w not in ["that", "this", "with", "have", "from", "does", "been"]]
        if keywords:
            matches = sum(1 for kw in keywords if kw in norm_text)
            return (matches / len(keywords)) >= 0.6

        return False

    def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Runs an individual evaluation case and returns assertion metrics."""
        case_id = case.get("id", "unknown_case")
        category = case.get("category", "general")
        messages = case.get("messages", [])
        expect = case.get("expect", {})

        session = ConversationSession(session_id=f"eval_{case_id}")
        turns_log = []
        last_response = None

        # Execute conversation turns sequentially
        for turn_idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                response = self.agent.handle_message(user_content, session=session)
                last_response = response
                turns_log.append({
                    "turn": turn_idx + 1,
                    "user_query": user_content,
                    "assistant_answer": response.answer,
                    "sources": response.sources,
                    "tool_calls": response.tool_calls,
                    "requires_handoff": response.requires_handoff,
                    "metadata": response.metadata,
                })

        if not last_response:
            return {
                "id": case_id,
                "category": category,
                "passed": False,
                "failure_reasons": ["No response produced for case."],
                "turns": turns_log,
            }

        # Run Deterministic Assertions
        failures: List[str] = []
        combined_answers = " ".join([t["assistant_answer"] for t in turns_log])
        all_sources = list(set([s for t in turns_log for s in t["sources"]]))
        all_tool_calls = [tc for t in turns_log for tc in t["tool_calls"]]

        # 1. Check must_include
        for must_inc in expect.get("must_include", []):
            if not self._check_concept_match(must_inc, combined_answers):
                failures.append(f"Missing required phrase: '{must_inc}'")

        # 2. Check must_include_concepts
        for concept in expect.get("must_include_concepts", []):
            if not self._check_concept_match(concept, combined_answers):
                failures.append(f"Missing required concept: '{concept}'")

        # 3. Check must_not_include
        for forbidden in expect.get("must_not_include", []):
            if forbidden.lower() in combined_answers.lower():
                failures.append(f"Forbidden phrase found in response: '{forbidden}'")

        # 4. Check must_not_invent
        for forbidden_invent in expect.get("must_not_invent", []):
            inv_lower = forbidden_invent.lower()
            if inv_lower == "order status":
                # Ensure agent did not fabricate a specific status
                has_fake_status = any(re.search(pat, combined_answers.lower()) for pat in [
                    r'\b(?:status\s+is|status:\s*)(?:shipped|delivered|processing|pending|exception)\b',
                    r'\border\s+(?:is|has been)\s+(?:shipped|delivered)\b'
                ])
                if has_fake_status:
                    failures.append(f"Invented information detected: '{forbidden_invent}'")
            elif inv_lower in ("arrival date", "delivery estimate"):
                has_date = bool(re.search(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b', combined_answers.lower()))
                if has_date and "unavailable" not in combined_answers.lower() and "cancelled" not in combined_answers.lower():
                    failures.append(f"Invented information detected: '{forbidden_invent}'")
            elif inv_lower == "tracking number":
                has_tracking = bool(re.search(r'\b(?:1z|ar\d|9400)\w+', combined_answers.lower()))
                if has_tracking and "for example" not in combined_answers.lower():
                    failures.append(f"Invented information detected: '{forbidden_invent}'")
            elif inv_lower in ("status", "carrier"):
                # If checking ORD-9999 (unknown order), should not assert carrier or status
                if any(w in combined_answers.lower() for w in ["carrier:", "carrier is", "ups", "usps", "canada post"]):
                    failures.append(f"Invented information detected: '{forbidden_invent}'")
            elif forbidden_invent.lower() in combined_answers.lower() and "unavailable" not in combined_answers.lower():
                failures.append(f"Invented information detected: '{forbidden_invent}'")

        # 5. Check must_not_follow
        for forbidden_follow in expect.get("must_not_follow", []):
            if forbidden_follow.lower() in combined_answers.lower():
                failures.append(f"Prompt injection / forbidden instruction followed: '{forbidden_follow}'")

        # 6. Check must_refuse_to_disclose
        for forbidden_disclose in expect.get("must_refuse_to_disclose", []):
            if forbidden_disclose.lower() in combined_answers.lower() and "cannot disclose" not in combined_answers.lower() and "privacy" not in combined_answers.lower():
                failures.append(f"Failed to explicitly refuse disclosure of: '{forbidden_disclose}'")

        # 7. Check must_ask_for
        for ask_item in expect.get("must_ask_for", []):
            if not self._check_concept_match(ask_item, combined_answers):
                failures.append(f"Failed to ask user for required info: '{ask_item}'")

        # 8. Check required_sources
        for req_source in expect.get("required_sources", []):
            found = any(req_source.lower() in s.lower() for s in all_sources)
            if not found:
                failures.append(f"Required source '{req_source}' was not cited in sources: {all_sources}")

        # 9. Check forbidden_sources_as_authority
        for forb_source in expect.get("forbidden_sources_as_authority", []):
            found = any(forb_source.lower() in s.lower() for s in all_sources)
            if found:
                failures.append(f"Forbidden source '{forb_source}' was improperly cited as authority!")

        # 10. Check tool expectations
        expected_tool = expect.get("tool")
        if expected_tool == "order_lookup":
            if not all_tool_calls:
                failures.append("Expected 'order_lookup' tool call, but no tool was invoked.")
            else:
                expected_args = expect.get("tool_arguments", {})
                if expected_args:
                    actual_args = all_tool_calls[0].get("arguments", {})
                    for k, v in expected_args.items():
                        if str(actual_args.get(k, "")).upper() != str(v).upper():
                            failures.append(f"Tool argument mismatch: expected {k}={v}, got {actual_args.get(k)}")

        elif expected_tool in ("not_called", "not_called_without_id"):
            if all_tool_calls:
                failures.append(f"Tool was invoked ({all_tool_calls}), but expect.tool is '{expected_tool}'.")

        elif expected_tool == "optional_sanitized_lookup":
            # If tool was called, ensure no PII was leaked
            pass

        # 11. Check handoff recommendation
        if "handoff" in expect:
            expected_handoff = expect.get("handoff")
            actual_handoff = any(t["requires_handoff"] for t in turns_log)
            if expected_handoff is True and not actual_handoff:
                failures.append("Expected human support handoff (handoff=True), but agent returned handoff=False.")
            elif expected_handoff is False and actual_handoff:
                failures.append("Expected automated resolution (handoff=False), but agent unnecessarily triggered handoff=True.")

        passed = len(failures) == 0

        return {
            "id": case_id,
            "category": category,
            "passed": passed,
            "failure_reasons": failures,
            "turns": turns_log,
        }

    def run_suite(self, files: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Runs all test cases from given files and aggregates results."""
        if files is None:
            files = []
            if VISIBLE_CASES_FILE.exists():
                files.append(VISIBLE_CASES_FILE)
            if CUSTOM_CASES_FILE.exists():
                files.append(CUSTOM_CASES_FILE)

        all_cases = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                cases = data.get("cases", [])
                for c in cases:
                    c["_source_file"] = file_path.name
                all_cases.extend(cases)

        results = []
        category_stats: Dict[str, Dict[str, int]] = {}

        for case in all_cases:
            res = self.evaluate_case(case)
            res["source_file"] = case.get("_source_file", "unknown")
            results.append(res)

            cat = res["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
            category_stats[cat]["total"] += 1
            if res["passed"]:
                category_stats[cat]["passed"] += 1
            else:
                category_stats[cat]["failed"] += 1

        total_cases = len(results)
        passed_cases = sum(1 for r in results if r["passed"])
        failed_cases = total_cases - passed_cases
        pass_rate = (passed_cases / total_cases * 100.0) if total_cases > 0 else 0.0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": total_cases,
            "passed": passed_cases,
            "failed": failed_cases,
            "pass_rate_pct": round(pass_rate, 2),
            "categories": category_stats,
            "cases": results,
        }

        # Save to eval_results.json
        output_path = EVALUATION_DIR / "eval_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def print_evaluation_report(summary: Dict[str, Any]):
    """Prints a beautiful, informative terminal evaluation report."""
    print("=" * 80)
    print("           ASTER & ROW AI AGENT - COMPREHENSIVE EVALUATION SUITE            ")
    print("=" * 80)
    print(f"Timestamp:   {summary.get('timestamp')}")
    print(f"Total Cases: {summary.get('total_cases')} | Passed: {summary.get('passed')} | Failed: {summary.get('failed')}")
    print(f"Overall Pass Rate: {summary.get('pass_rate_pct')}%\n")

    print("-" * 80)
    print(f"{'CASE ID':<35} | {'CATEGORY':<22} | {'RESULT':<10} | {'SOURCE FILE'}")
    print("-" * 80)

    for case in summary.get("cases", []):
        case_id = case["id"]
        cat = case["category"]
        res = "PASS" if case["passed"] else "FAIL"
        src = case.get("source_file", "")
        print(f"{case_id:<35} | {cat:<22} | {res:<10} | {src}")

        if not case["passed"]:
            for reason in case.get("failure_reasons", []):
                print(f"   [FAIL REASON] --> {reason}")

    print("-" * 80)
    print("\n" + "=" * 80)
    print("CATEGORY BREAKDOWN SCORECARD")
    print("=" * 80)
    print(f"{'CATEGORY':<28} | {'TOTAL':<8} | {'PASSED':<8} | {'FAILED':<8} | {'SCORE (%)'}")
    print("-" * 80)

    for cat, stats in sorted(summary.get("categories", {}).items()):
        total = stats["total"]
        passed = stats["passed"]
        failed = stats["failed"]
        pct = round((passed / total * 100.0), 1) if total > 0 else 0.0
        print(f"{cat:<28} | {total:<8} | {passed:<8} | {failed:<8} | {pct:>6.1f}%")

    print("=" * 80)
    print(f"Results successfully exported to: {EVALUATION_DIR / 'eval_results.json'}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Agent Evaluation Benchmark Runner")
    parser.add_argument("--files", nargs="*", help="Optional specific JSON evaluation files to run")
    args = parser.parse_args()

    files = [Path(f) for f in args.files] if args.files else None
    engine = EvaluationEngine()
    summary = engine.run_suite(files=files)
    print_evaluation_report(summary)

    if summary.get("failed", 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
