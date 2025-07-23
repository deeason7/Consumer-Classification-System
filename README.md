# Consumer Complaint Sentiment Insights and Classification System


An end-to-end NLP pipeline that classifies the sentiment of consumer financial complaints using a hybrid deep learning model, accessible via an interactive command-line interface.

---

## 1. Context & Motivation

The **Consumer Financial Protection Bureau (CFPB)** collects thousands of consumer complaints across financial services. Understanding the emotional intensity of these complaint narratives (Neutral, Negative, Extreme Negative) enables us to:
* Detect systemic issues in financial services.
* Improve customer-service strategies at institutions.
* Inform policy decisions and enforcement priorities.
* Prioritize high-risk cases for investigation.

---

## 2. Problem Statement

How can we leverage NLP and deep learning to build a scalable, production-ready pipeline that classifies consumer-complaint narratives into three sentiment levels:
* Neutral
* Negative
* Extreme Negative

---

## 3. Key Features

* **Hybrid Deep Learning Model**: The system's core is a BiLSTM model that processes both the text of a complaint and structured metadata (like text length and keyword flags) for more nuanced predictions.
* **Interactive CLI**: A user-friendly Command-Line Interface (`app.py`) allows for real-time predictions by simply typing or pasting a complaint narrative.
* **Automated Logging**: Every prediction is automatically timestamped and saved to a local SQLite database (`storage.py`), creating a verifiable log of model activity.
* **Modular & Scalable Design**: The code is cleanly separated into modules for loading the model (`model.py`), handling predictions (`predictor.py`), and managing storage (`storage.py`), making it easy to maintain and extend.

---

## 4. Hypotheses Explored

The initial analysis was guided by several hypotheses to uncover deeper insights:
* **H1**: Products like *Credit Reporting* and *Debt Collection* exhibit higher Extreme-Negative rates.
* **H2**: Longer narratives correlate with higher emotional intensity.
* **H3**: Certain companies show systemic patterns of Extreme-Negative sentiment.
* **H4**: Trigger keywords (e.g., “fraud”, “lawsuit”) strongly correlate with Extreme-Negative sentiment.
* **H5**: Timely response has only a minor effect on sentiment compared to the quality of the experience.

---

## 5. Tech Stack

| Area | Technologies |
| :--- | :--- |
| **Language** | Python 3.8 (via Pyenv), Jupyter Notebooks |
| **Data & Modeling** | Pandas, NumPy, TensorFlow / Keras (BiLSTM), Scikit-learn |
| **NLP Preprocessing** | NLTK, SpaCy, TextBlob |
| **Data Storage** | SQLite |
| **Version Control** | Git & GitHub |

---

## 6. Project Workflow & Status

| # | Objective | Status |
| :- | :--- | :--- |
| 1 | Load & explore raw dataset; perform basic structural cleaning | Complete |
| 2 | Clean & normalize complaint text; engineer initial “weak” sentiment labels | Complete |
| 3 | Exploratory Data Analysis on product, company, sentiment & interaction features | Complete |
| 4 | Train baseline BiLSTM deep-learning model for sentiment classification | Complete |
| 5 | Test & demonstrate the trained BiLSTM model (CLI Application) | Complete |

---

## 7. Installation

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/your-username/consumer-complaint-sentiment.git](https://github.com/your-username/consumer-complaint-sentiment.git)
    cd consumer-complaint-sentiment
    ```

2.  **Create and Activate a Virtual Environment**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a file named `.env` in the root of the project directory. This file stores the paths to your model and artifacts. Add the following content to it, replacing the paths with the actual locations of your files:
    ```env
    MODEL_PATH="models/sentiment_model.keras"
    TOKENIZER_PATH="outputs/tokenizer_sentiment.pkl"
    LABEL_ENCODER_PATH="outputs/label_encoder_sentiment.pkl"
    DB_PATH="deployment/predictions.db"
    ```

---

## 8. Usage

The primary way to interact with the model is through the command-line application.

1.  **Navigate to the project directory** and ensure your virtual environment is activated.

2.  **Run the application:**
    ```sh
    python deployment/app.py
    ```

3.  **Interact with the CLI:**
    The application will load the model and artifacts, then present you with an interactive prompt. You can type or paste a complaint narrative and press Enter to get a real-time sentiment prediction.

    
