from typing import Dict, Any

def safe_score(score: float) -> float:
    """Ensures score is strictly within (0.01, 0.99) range."""
    try:
        s = float(score)
        return max(0.01, min(0.99, s))
    except (ValueError, TypeError):
        return 0.5

def grade_easy(data: dict, *args, **kwargs) -> float:
    """Validator passes state dict here. We extract current_score."""
    if not isinstance(data, dict): return safe_score(0.5)
    return safe_score(data.get("current_score", 0.5))

def grade_medium(data: dict, *args, **kwargs) -> float:
    if not isinstance(data, dict): return safe_score(0.5)
    return safe_score(data.get("current_score", 0.5))

def grade_hard(data: dict, *args, **kwargs) -> float:
    if not isinstance(data, dict): return safe_score(0.5)
    return safe_score(data.get("current_score", 0.5))
