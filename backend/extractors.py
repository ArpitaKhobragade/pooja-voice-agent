import re
from typing import Any

def extract_slot(slot_name: str, transcript: str) -> tuple[Any | None, float]:
    """
    Deterministically extract a value for the given slot from the transcript.
    Returns (extracted_value, confidence).
    If the slot is not supported or confidence is too low, returns (None, 0.0).
    """
    if not transcript or not transcript.strip():
        return None, 0.0
        
    if slot_name == "good_time_confirmed":
        return _extract_good_time_confirmed(transcript.strip())
    elif slot_name == "project":
        return _extract_project(transcript.strip())
    elif slot_name == "timeline":
        return _extract_timeline(transcript.strip())
        
    return None, 0.0

def _extract_good_time_confirmed(text: str) -> tuple[str | None, float]:
    """
    Extract whether the user confirmed it's a good time to talk.
    Returns ("yes" | "no", confidence).
    """
    text = text.lower()
    
    # Strip punctuation for cleaner matching
    text_clean = re.sub(r'[^\w\s]', '', text)
    
    # Affirmative patterns
    yes_patterns = [
        r'^(yes|yeah|yep|sure|yup|ok|okay)$',
        r'^(it is|it sure is|yes it is)$',
        r'^(go ahead|tell me|im listening)$',
        r'^(of course|absolutely|definitely)$'
    ]
    
    # Negative patterns
    no_patterns = [
        r'^(no|nope|nah)$',
        r'^(not right now|not a good time|busy|im busy|driving|im driving|call back|later)$',
        r'^(call me later|call later|bad time)$'
    ]
    
    # Check exact matches first (Highest confidence)
    for p in yes_patterns:
        if re.match(p, text_clean):
            return "yes", 1.0
            
    for p in no_patterns:
        if re.match(p, text_clean):
            return "no", 1.0
            
    # Check for strong substrings if no exact match (Medium-high confidence)
    if "yes" in text_clean.split() or "sure" in text_clean.split():
        return "yes", 0.90
        
    if "no" in text_clean.split() or "busy" in text_clean.split() or "driving" in text_clean.split():
        return "no", 0.90
        
    return None, 0.0

def _extract_project(text: str) -> tuple[str | None, float]:
    """
    Extract the renovation project type.
    Since projects are free-form, we look for common keywords or very short answers.
    """
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()
    
    # If the user gives a very short answer (1-4 words) that includes common project nouns,
    # we can confidently extract the whole phrase as the project.
    project_keywords = {"kitchen", "bathroom", "bedroom", "living", "room", "house", "flat", "apartment", "office", "interior", "exterior", "renovation", "remodel"}
    
    if len(words) <= 5 and any(kw in words for kw in project_keywords):
        return text.strip(), 0.90
        
    # If it's a longer sentence, but strongly focuses on one keyword
    # we can extract just the keyword phase.
    for kw in ["kitchen", "bathroom", "bedroom", "living room", "full house", "entire flat"]:
        if kw in text_clean:
            return kw, 0.80
            
    return None, 0.0

def _extract_timeline(text: str) -> tuple[str | None, float]:
    """
    Extract the timeline for the project.
    """
    text_clean = re.sub(r'[^\w\s]', '', text.lower())
    words = text_clean.split()
    
    timeline_keywords = {"month", "months", "week", "weeks", "year", "soon", "immediately", "urgent", "now", "days"}
    
    if len(words) <= 5 and any(kw in words for kw in timeline_keywords):
        return text.strip(), 0.90
        
    # Check for common phrases in longer text
    patterns = [
        r'(next (month|week|year))',
        r'(in \d+ (months|weeks|days))',
        r'(asap|as soon as possible|immediately|right away)',
        r'(no rush|sometime later)'
    ]
    
    for p in patterns:
        match = re.search(p, text_clean)
        if match:
            return match.group(0), 0.85
            
    return None, 0.0
