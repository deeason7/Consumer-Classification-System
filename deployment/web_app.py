# deployment/web_app.py
import os
import json
from flask import Flask, render_template, request, redirect, url_for
from dotenv import load_dotenv


from deployment.model import ModelLoader
from deployment.predictor import Predictor
from deployment.storage import Storage

# Initialization
load_dotenv()
MODEL_PATH = os.environ.get("MODEL_PATH")
DB_PATH = os.environ.get("DB_PATH")

try:
    # Load model, artifacts, and initialize components
    model = ModelLoader(MODEL_PATH).load()
    predictor = Predictor(model=model)
    storage = Storage(db_path=DB_PATH)

    # Load the feature config for UI dropdowns
    with open("../outputs/feature_config.json", 'r') as f:
        feature_config = json.load(f)

    ALL_PRODUCTS = feature_config.get("all_products", [])
    ALL_COMPANIES = feature_config.get("all_companies", [])

    print("All components initialized successfully for web app.")

except Exception as e:
    print(f"Error initializing web app components: {e}")
    predictor = None
    storage = None

# Create the Flask web application
app = Flask(__name__)

#CUSTOMER FACING ROUTES

@app.route('/')
def index():
    """Renders the main page with the complaint form."""
    return render_template('index.html', products=ALL_PRODUCTS, companies=ALL_COMPANIES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    if not predictor:
        return "Error: Application not initialized. Please check server logs.", 500

    try:
        # Get data from the form
        text = request.form['text']
        product = request.form['product_other'] if request.form['product'] == 'Other' else request.form['product']
        company = request.form['company_other'] if request.form['company'] == 'Other' else request.form['company']
        timely = request.form['timely']

        # Get prediction result
        result = predictor.predict(text, product, company, timely)

        # Log the complaint to the database and get the new ID
        complaint_id = storage.submit_complaint(
            text=text,
            product=product,
            company=company,
            label=result['label'],
            confidence=result['confidence'],
        )

        # Redirect to a success page showing the new ID
        return redirect(url_for('submission_success', complaint_id=complaint_id))
    except Exception as e:
        return f"An error occurred during prediction: {e}", 500

@app.route('/success/<int:complaint_id>')
def submission_success(complaint_id):
    """Displays a confirmation page with the complaint ID."""
    return render_template('submission_success.html', complaint_id=complaint_id)

@app.route('/status', methods=['GET', 'POST'])
def status_check():
    """Handles the status check page and lookup logic."""
    if request.method == 'POST':
        complaint_id = request.form.get('complaint_id')

        if not complaint_id:
            return render_template('status_check.html', error="Please enter a Complaint ID.")

        complaint = storage.get_complaint_by_id(int(complaint_id))
        return render_template('status_display.html', complaint=complaint)

    # For GET requests, just show the lookup form
    return render_template('status_check.html')

#AGENT FACING ROUTES
@app.route('/dashboard')
def dashboard():
    """
    Renders the agent dashboard with complaints sorted by priority.
    """
    if not  storage:
        return "Error: Storage not initialized.", 500

    complaints = storage.get_all_complaints_by_priority()
    return render_template('dashboard.html', complaints=complaints)

@app.route('/complaint/<int:complaint_id>', methods=['GET', 'POST'])
def complaint_detail(complaint_id):
    """Handles  viewing and updating a single complaint."""
    if not storage:
        return "Error: Storage not initialized.", 500

    if request.method == 'POST':
        new_status = request.form['status']
        agent_notes = request.form['agent_notes']
        storage.update_complaint_status(complaint_id, new_status, agent_notes)
        return redirect(url_for('dashboard'))

    # For GET requests, show the detail page
    complaint = storage.get_complaint_by_id(complaint_id)
    return render_template('complaint_detail.html', complaint=complaint)

def run():
    """Runs the Flask application."""
    app.run(debug=True, port=5001, use_reloader=False)