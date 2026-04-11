from typing import Dict, Any

def safe_score(score: float) -> float:
    """Validator requires score strictly within (0, 1)."""
    try:
        s = float(score)
        # Use simple clamping precisely as requested
        return max(0.01, min(0.99, s))
    except (ValueError, TypeError):
        return 0.5

def grade_easy(data: dict, *args, **kwargs) -> float:
    """OpenEnv passes the state dict here. We extract current_score."""
    print("GRADER INPUT (EASY):", data)
    if not isinstance(data, dict): return safe_score(0.5)
    return safe_score(data.get("current_score", 0.5))

def grade_medium(data: dict, *args, **kwargs) -> float:
    print("GRADER INPUT (MEDIUM):", data)
    if not isinstance(data, dict): return safe_score(0.5)
    return safe_score(data.get("current_score", 0.5))

def grade_hard(data: dict, *args, **kwargs) -> float:
    print("GRADER INPUT (HARD):", data)
    if not isinstance(data, dict): return safe_score(0.5)
    return safe_score(data.get("current_score", 0.5))
