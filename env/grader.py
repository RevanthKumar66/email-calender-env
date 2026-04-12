from typing import Dict, Any

def safe_score(score: float) -> float:
    """Ensures score is strictly within (0.01, 0.99) range as required by platform."""
    try:
        s = float(score)
        return max(0.01, min(0.99, s))
    except (ValueError, TypeError):
        return 0.5

def grade_easy(data: Any, *args, **kwargs) -> float:
    """Extracts score from task state or returns a valid default."""
    if isinstance(data, dict) and "current_score" in data:
        return safe_score(data["current_score"])
    return safe_score(0.1)

def grade_medium(data: Any, *args, **kwargs) -> float:
    if isinstance(data, dict) and "current_score" in data:
        return safe_score(data["current_score"])
    return safe_score(0.1)

def grade_hard(data: Any, *args, **kwargs) -> float:
    if isinstance(data, dict) and "current_score" in data:
        return safe_score(data["current_score"])
    return safe_score(0.1)
