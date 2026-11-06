"""Build a submit_answer payload that the oracle would send. Used by dry-run and tests."""

from fdagym.gold_cite import citations_from_outputs, gold_answer, gold_submit

__all__ = ["citations_from_outputs", "gold_answer", "gold_submit"]
