# deployment/feature_engineer.py
import json
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class FeatureEngineer:
    """
    Handles the creation of the structured feature vector for live prediction
    by loading pre-computed artifacts from the training pipeline.
    """

    def __init__(self,
                 product_map_path: str = "../outputs/product_dispute_map.pkl",
                 company_map_path: str = "../outputs/company_dispute_map.pkl",
                 config_path: str = "../outputs/feature_config.json"):

        with open(product_map_path, 'rb') as f:
            self.product_map = pickle.load(f)
        with open(company_map_path, 'rb') as f:
            self.company_map = pickle.load(f)

        with open(config_path, 'r') as f:
            config = json.load(f)
            self.global_mean = config['global_mean_dispute_rate']
            self.keywords = config.get('extreme_negative_keywords', []) + config.get('negative_keywords', [])

    def _find_best_match(self, user_input: str, mapping_dict: dict) -> str:
        """
        Finds the best match for a user's input in a dictionary's keys.
        Tries for a case-insensitive match first, then a partial match.
        """
        user_input_lower = user_input.lower()

        # Try for an exact case-insensitive match
        for key in mapping_dict:
            if key.lower() == user_input_lower:
                return key

        # If no exact match, try for a partial match (substring)
        for key in mapping_dict:
            if user_input_lower in key.lower():
                return key

        # If no match found, return the original input
        return user_input

    def engineer_features(self, text: str, product: str, company: str, timely_response: str) -> np.ndarray:
        """
        Creates the structured feature vector from raw inputs using robust matching.
        """
        #  Find the best matching keys for product and company
        product_key = self._find_best_match(product, self.product_map)
        company_key = self._find_best_match(company, self.company_map)

        #  Engineer features using the matched keys
        text_length = float(len(text.split()))
        timely_response_binary = 1.0 if timely_response.lower() == 'yes' else 0.0
        product_dispute_rate = float(self.product_map.get(product_key, self.global_mean))
        company_dispute_rate = float(self.company_map.get(company_key, self.global_mean))
        keyword_flag = 1.0 if any(kw in text.lower() for kw in self.keywords) else 0.0

        # Assemble the final feature vector
        struct_vec = np.array([[
            text_length,
            timely_response_binary,
            product_dispute_rate,
            company_dispute_rate,
            keyword_flag
        ]], dtype=np.float32)

        return struct_vec