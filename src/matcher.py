import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ValuesMatcher:
    """
    Takes a dictionary of profile embeddings and finds the most
    compatible matches for any given user based on values similarity.
    """
    
    def __init__(self, embeddings, profiles):
        self.embeddings = embeddings
        self.profiles = {p["id"]: p for p in profiles}
        self.ids = list(embeddings.keys())
        self.matrix = np.array(list(embeddings.values()))
    
    def get_similarity_score(self, id_a, id_b):
        """
        Returns the cosine similarity score between two profiles.
        Score ranges from 0 to 1. Higher means more compatible.
        """
        vec_a = self.embeddings[id_a].reshape(1, -1)
        vec_b = self.embeddings[id_b].reshape(1, -1)
        return float(cosine_similarity(vec_a, vec_b)[0][0])
    
    def get_top_matches(self, user_id, top_n=3):
        """
        Returns the top N most compatible profiles for a given user,
        excluding the user themselves.
        """
        user_vec = self.embeddings[user_id].reshape(1, -1)
        scores = cosine_similarity(user_vec, self.matrix)[0]
        
        ranked = sorted(
            zip(self.ids, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        matches = []
        for match_id, score in ranked:
            if match_id != user_id:
                matches.append({
                    "id": match_id,
                    "name": self.profiles[match_id]["name"],
                    "score": round(float(score), 4),
                    "values": self.profiles[match_id]["values"],
                    "goals": self.profiles[match_id]["goals"]
                })
        
        return matches[:top_n]
    
    def get_full_similarity_matrix(self):
        """
        Returns a matrix of similarity scores between all profiles.
        Useful for visualization in the notebook.
        """
        scores = cosine_similarity(self.matrix, self.matrix)
        return scores