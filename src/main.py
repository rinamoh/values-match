import json
import sys
from src.embedder import ProfileEmbedder
from src.matcher import ValuesMatcher


def main():
    profiles = ProfileEmbedder.load_profiles("data/profiles.json")
    embedder = ProfileEmbedder()
    embeddings = embedder.embed_all_profiles(profiles)
    matcher = ValuesMatcher(embeddings, profiles)
    
    # Accept optional user ID from command line
    # Usage: python -m src.main user_007
    # Default: runs demo for three sample users
    if len(sys.argv) > 1:
        demo_users = [sys.argv[1]]
    else:
        demo_users = ["user_001", "user_003", "user_005"]
    
    for user_id in demo_users:
        profile = next(
            (p for p in profiles if p["id"] == user_id), None
        )
        if not profile:
            print(f"User {user_id} not found. "
                  f"Available IDs: user_001 through user_015")
            continue
            
        print(f"\n{'='*60}")
        print(f"Finding matches for: {profile['name']} ({user_id})")
        print(f"Their values:")
        for v in profile["values"]:
            print(f"  - {v}")
        
        print(f"\nTop 3 most compatible matches:")
        matches = matcher.get_top_matches(user_id, top_n=3)
        
        for i, match in enumerate(matches, 1):
            print(f"\n  Match {i}: {match['name']} "
                  f"(compatibility score: {match['score']})")
            print(f"  Their values:")
            for v in match["values"]:
                print(f"    - {v}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()