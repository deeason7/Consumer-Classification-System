# deployment/predictor.py
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv

from deployment.feature_engineer import FeatureEngineer

class Predictor:
    """
    Encapsulates artifact loading, preprocessing, and model inference.
    """
    def __init__(self,
                 model,
                 tokenizer_path: str = None,
                 label_encoder_path: str = None,
                 max_len: int = 250):
        
        self.model = model
        self.max_len = max_len

        # Initialize the feature engineer to handle structured data
        self.feature_engineer = FeatureEngineer()

        # Load tokenizer and label encoder from environment variables
        load_dotenv()
        tokenizer_path = tokenizer_path or os.environ.get("TOKENIZER_PATH")
        label_encoder_path = label_encoder_path or os.environ.get("LABEL_ENCODER_PATH")

        if not tokenizer_path or not label_encoder_path:
            raise ValueError("Tokenizer or Label Encoder path not set in environment.")

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def _preprocess_text(self, text: str) -> np.ndarray:
        """Tokenizes and pads a single text string."""
        seq = self.tokenizer.texts_to_sequences([text.lower()])
        return pad_sequences(seq, maxlen=self.max_len)

    def predict(self, text: str, product: str, company: str, timely_response: str) -> dict:
        """
        Runs a prediction on a full set of inputs.
        """
        # Preprocess the text input
        padded_seq = self._preprocess_text(text)
        
        # Engineer the structured feature vector using the dedicated class
        struct_vec = self.feature_engineer.engineer_features(text, product, company, timely_response)

        # Predict class probabilities using both inputs
        probs = self.model.predict([padded_seq, struct_vec], verbose=0).flatten()

        # Determine the final prediction and confidence
        pred_index = int(np.argmax(probs))
        confidence = float(probs[pred_index])
        label = self.label_encoder.inverse_transform([pred_index])[0]

        # Package the results, including the features used for transparency
        features_used = {
            'text_length': struct_vec[0, 0],
            'timely_response_binary': int(struct_vec[0, 1]),
            'product_dispute_rate': round(struct_vec[0, 2], 4),
            'company_dispute_rate': round(struct_vec[0, 3], 4),
            'keyword_flag': int(struct_vec[0, 4])
        }

        return {'label': label, 'confidence': confidence, 'structured_features': features_used}