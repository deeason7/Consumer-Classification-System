# deployment/app.py
import os
import datetime
from dotenv import load_dotenv

from deployment import web_app
from deployment.model import ModelLoader
from deployment.predictor import Predictor
from deployment.storage import Storage


def run_cli():
    """Contains the logic for the command-line interface."""
    load_dotenv()

    MODEL_PATH = os.environ.get("MODEL_PATH")
    DB_PATH = os.environ.get("DB_PATH")

    try:
        model = ModelLoader(MODEL_PATH).load()
        predictor = Predictor(model=model)
        storage = Storage(db_path=DB_PATH)
        print("CLI Components Initialized.\n")
    except Exception as err:
        print(f"Initialization Error: {err}")
        return

    print("Consumer Complaint Sentiment CLI")
    print("Enter the details for a new complaint. Type 'exit' to quit.\n")

    while True:
        text = input("Enter complaint narrative: ").strip()
        if text.lower() in ("exit", "quit"): break

        product = input("Enter product (e.g., 'Mortgage'): ").strip()
        company = input("Enter company (e.g., 'Wells Fargo & Company'): ").strip()
        timely = input("Was the response timely? (yes/no): ").strip()

        if not all([text, product, company, timely]):
            print("\nAll fields are required. Please try again.\n")
            continue

        # Get the prediction result
        result = predictor.predict(text, product, company, timely)

        print("\nPrediction Result")
        print(f"  Sentiment Label:  {result['label']}")
        print(f"  Confidence Score: {result['confidence']:.2f}")

        # Log the complaint
        timestamp = datetime.datetime.now().isoformat()
        try:
            complaint_id = storage.submit_complaint(
                text=text,
                product=product,
                company=company,
                label=result['label'],
                confidence=result['confidence']
            )
            print(f"\n Logged prediction with ID {complaint_id} at {timestamp}\n")
        except Exception as err:
            print(f"\n Failed to log prediction: {err}\n")


def run_web_interface():
    """Launches the Flask web application."""
    print(" Launching Flask web server...")
    print("Navigate to http://127.0.0.1:5001 in your web browser.")
    web_app.run()


def main():
    """Presents a menu to the user to choose the interface."""
    while True:
        print("\n Main Menu ")
        print("1. Launch Web Interface")
        print("2. Launch Command-Line Interface (CLI)")
        print("3. Exit")
        choice = input("Please choose an option (1, 2, or 3): ").strip()

        if choice == '1':
            run_web_interface()
            break
        elif choice == '2':
            run_cli()
            break
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print("Exiting. Goodbye!")


if __name__ == "__main__":
    main()