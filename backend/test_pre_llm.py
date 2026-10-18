import asyncio
from extractors import extract_slot
from conversation_manager import ConversationManager

def test_extraction():
    cm = ConversationManager("1234567890")
    
    # 1. Test good_time_confirmed extraction
    val, conf = extract_slot("good_time_confirmed", "Yes, it is.")
    print(f"Extraction for 'Yes, it is.': {val} (conf={conf})")
    assert val == "yes" and conf >= 0.9

    val, conf = extract_slot("good_time_confirmed", "Not right now, I'm driving.")
    print(f"Extraction for 'Not right now...': {val} (conf={conf})")
    assert val == "no" and conf >= 0.9

    val, conf = extract_slot("good_time_confirmed", "I guess so.")
    print(f"Extraction for 'I guess so.': {val} (conf={conf})")
    assert val == None

    val, conf = extract_slot("project", "kitchen")
    print(f"Extraction for project (unsupported): {val} (conf={conf})")
    assert val == None
    
    print("All extractor tests passed!")

if __name__ == "__main__":
    test_extraction()
