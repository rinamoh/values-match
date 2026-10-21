import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


class ProfileEmbedder:
    """
    Converts user profiles into numerical embeddings using a 
    sentence transformer model. Similar values will produce 
    similar embeddings even when worded differently.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        print("Model loaded successfully")
    
    def profile_to_text(self, profile):
        """
        Combines all values and goals into a single text string.
        This is what gets embedded.
        """
        all_statements = profile["values"] + profile["goals"]
        return " ".join(all_statements)
    
    def embed_profile(self, profile):
        """
        Returns a single embedding vector for one profile.
        """
        text = self.profile_to_text(profile)
        return np.array(self.model.embed_query(text))
    
    def embed_all_profiles(self, profiles):
        """
        Returns a dictionary mapping user IDs to their embeddings.
        """
        print(f"Embedding {len(profiles)} profiles...")
        embeddings = {}
        for profile in profiles:
            embeddings[profile["id"]] = self.embed_profile(profile)
            print(f"  Embedded {profile['id']} ({profile['name']})")
        print("All profiles embedded successfully")
        return embeddings
    
    @staticmethod
    def load_profiles(filepath):
        """
        Loads profiles from a JSON file.
        """
        with open(filepath, "r") as f:
            return json.load(f)