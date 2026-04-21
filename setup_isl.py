import os
import json

ASSETS_DIR = "assets"
VIDEO_DIR = os.path.join(ASSETS_DIR, "videos", "in-myorg-1")

VOCAB_LIST = [
    "hello", "goodbye", "yes", "no", "please", "sorry", "thank", "you",
    "me", "they", "we", "mother", "father", "boy", "girl", "baby",
    "name", "what", "where", "when", "why", "who", "how", "eat",
    "drink", "sleep", "go", "come", "want", "help", "love", "see",
    "stop", "good", "bad", "happy", "sad", "hot", "cold", "today",
    "tomorrow", "yesterday", "now", "home", "school", "book", "food",
    "water", "bathroom", "friend"
]

def generate_vocab_json():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    mapping = []
    
    for word in VOCAB_LIST:
        # Basic token mapping (adding ing/s hooks for testing, simple heuristic)
        tokens = [word]
        if word in ["eat", "drink", "sleep", "go", "come", "help", "see"]:
            if word == "see":
                tokens.append("seeing")
            elif word == "come":
                tokens.append("coming")
            else:
                tokens.append(word + "ing")
                
        mapping.append({
            "label": f"in-myorg-1_{word}",
            "token": {
                "en": tokens
            }
        })
        
    dataset = [
        {
            "country": "in",
            "organization": "myorg",
            "part_number": "1",
            "url": "",
            "description": "50 Most Common Custom ISL Vocabulary",
            "mapping": mapping
        }
    ]
    
    with open(os.path.join(ASSETS_DIR, "in-dictionary-mapping.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print("Created in-dictionary-mapping.json with 50 words")

def generate_dummy_videos():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    for word in VOCAB_LIST:
        filename = os.path.join(VIDEO_DIR, f"in-myorg-1_{word}.mp4")
        with open(filename, "wb") as f:
            f.write(b"empty dummy mp4 structure")
        
    print(f"Created {len(VOCAB_LIST)} dummy mp4 videos in {VIDEO_DIR}.")

if __name__ == "__main__":
    generate_vocab_json()
    generate_dummy_videos()
