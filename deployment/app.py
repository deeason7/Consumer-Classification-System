# deployment/app.py
import os
import datetime
from dotenv import load_dotenv

from deployment.model import ModelLoader
from deployment.predictor import Predictor
from deployment.storage import Storage

def main():
    load_dotenv()

    # Fetch paths from environment variables
    MODEL_PATH = os.environ.get("MODEL_PATH")
    DB_PATH = os.environ.get("DB_PATH")

    # Initialize components
    try:
        model = ModelLoader(MODEL_PATH).load()
        print("Model loaded successfully.")
        
        predictor = Predictor(model=model)
        print("Predictor initialized successfully.")
        
        storage = Storage(db_path=DB_PATH)
        print(f"SQLite database ready at: {DB_PATH}\n")
    except Exception as err:
        print(f"Initialization Error: {err}")
        return

    # Interactive loop for user input
    print("Consumer Complaint Sentiment CLI")
    print("Enter the details for a new complaint. Type 'exit' to quit.\n")

    while True:
        text = input("Enter complaint narrative: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        
        product = input("Enter product (e.g., 'Mortgage', 'Credit card'): ").strip()
        company = input("Enter company (e.g., 'Wells Fargo & Company'): ").strip()
        timely = input("Was the response timely? (yes/no): ").strip()

        if not all([text, product, company, timely]):
            print("\n All fields are required. Please try again.\n")
            continue

        # Run prediction with all inputs
        try:
            result = predictor.predict(text, product, company, timely)
        except Exception as err:
            print(f" Prediction error: {err}")
            continue

        # Display results clearly
        print("\nPrediction Result")
        print(f"  Sentiment Label:  {result['label']}")
        print(f"  Confidence Score: {result['confidence']:.2f}")
        print("\n  Features Used:")
        for name, val in result['structured_features'].items():
            print(f"    - {name}: {val}")
        
        # Log the prediction to the database
        timestamp = datetime.datetime.now().isoformat()
        try:
            storage.log(
                timestamp=timestamp, 
                text=text, 
                label=result['label'], 
                confidence=result['confidence'],
                # Store features as a string for logging
                features=str(result['structured_features'])
            )
            print(f"\n Logged prediction at {timestamp}\n")
        except Exception as err:
            print(f"\n Failed to log prediction: {err}\n")

    print("Exiting. Goodbye!")

if __name__ == "__main__":
    main()