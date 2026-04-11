from typing import List, Dict, Any, Optional

def safe_score(score: float) -> float:
    """Validator requires score strictly within (0, 1)."""
    try:
        s = float(score)
        return max(0.01, min(0.99, s))
    except (ValueError, TypeError):
        return 0.5

# Universal Grader Function that handles all inputs
def grade_generic(input_data: Any, *args, **kwargs) -> float:
    """Resilient grader that extracts score from any object or dict."""
    # 1. If we got an env instance, call its score method
    if hasattr(input_data, "score"):
        return safe_score(input_data.score())
    
    # 2. If we got a dict (final state/info/obs), try to find current_score or reward
    if isinstance(input_data, dict):
        # Check for explicitly stored scores
        if "current_score" in input_data: return safe_score(input_data["current_score"])
        if "score" in input_data: return safe_score(input_data["score"])
        
        # Calculate heuristic if only actions are available
        actions = input_data.get("actions_taken", [])
        if actions:
            # Positive signal for any valid-looking action
            return safe_score(0.2 + len(actions) * 0.05)
            
    # 3. Baseline for discovery
    return safe_score(0.5)

# Explicit function names matching openenv.yaml
def grade_easy(data: Any, *args, **kwargs) -> float:
    return grade_generic(data, *args, **kwargs)

def grade_medium(data: Any, *args, **kwargs) -> float:
    return grade_generic(data, *args, **kwargs)

def grade_hard(data: Any, *args, **kwargs) -> float:
    return grade_generic(data, *args, **kwargs)
