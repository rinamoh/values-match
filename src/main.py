import json
from src.embedder import ProfileEmbedder
from src.matcher import ValuesMatcher


def main():
    # Load profiles
    profiles = ProfileEmbedder.load_profiles("data/profiles.json")
    print(f"\nLoaded {len(profiles)} profiles\n")
    
    # Create embeddings
    embedder = ProfileEmbedder()
    embeddings = embedder.embed_all_profiles(profiles)
    
    # Initialize matcher
    matcher = ValuesMatcher(embeddings, profiles)
    
    # Demo: find matches for first three users
    demo_users = ["user_001", "user_003", "user_005"]
    
    for user_id in demo_users:
        profile = next(p for p in profiles if p["id"] == user_id)
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
    print("\nDone.")


if __name__ == "__main__":
    main()