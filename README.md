# Consumer Complaint Sentiment Insights and Classification System

An end-to-end NLP pipeline that classifies the sentiment of consumer financial complaints using a hybrid deep learning model optimized with hyperparameter tuning and served via an interactive command-line interface.

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

* **Hybrid Deep Learning Model**: The system's core is a BiLSTM model that processes both the complaint text and structured metadata (like text length and keyword flags) for more nuanced predictions.
* **Hyperparameter Tuning**: Utilizes **Keras Tuner** to systematically find the optimal model architecture and parameters, significantly boosting performance over a baseline model.
* **Interactive CLI**: A user-friendly Command-Line Interface (`app.py`) allows for real-time predictions by simply typing or pasting a complaint narrative.
* **Automated Logging**: Every prediction is automatically timestamped and saved to a local **SQLite** database (`storage.py`), creating a verifiable log of model activity.
* **Modular & Scalable Design**: The code is cleanly separated into modules for data processing,and a dedicated `deployment/` package for inference, making it easy to maintain and extend.

---

## 4. Hypotheses Explored

The initial analysis was guided by several hypotheses to uncover deeper insights:
* **H1: (Confirmed)** Products like *Credit Reporting* and *Debt Collection* exhibit higher Extreme-Negative rates.
* **H2: (Supported)** Longer narratives correlate with higher emotional intensity.
* **H3: (Confirmed)** Certain companies show systemic patterns of Extreme-Negative sentiment.
* **H4: (Confirmed)** Trigger keywords (e.g., “fraud”, “lawsuit”) strongly correlate with Extreme-Negative sentiment.
* **H5: (Rejected)** A company's timely response has only a minor effect on sentiment.

---

## 5. Tech Stack

| Area                      | Technologies                                               |
| :------------------------ | :--------------------------------------------------------- |
| **Language** | Python 3, Jupyter Notebooks                                |
| **Data & Modeling** | Pandas, NumPy, TensorFlow / Keras (BiLSTM), Scikit-learn   |
| **Hyperparameter Tuning** | Keras Tuner                                                |
| **NLP Preprocessing** | NLTK, TextBlob                                             |
| **Deployment & Storage** | CLI Application, SQLite, Dotenv                            |
| **Version Control** | Git & GitHub                                               |

---

## 6. Project Workflow & Status

| #  | Objective                                                                     | Status    |
| :--| :---------------------------------------------------------------------------- | :-------- |
| 1  | Load & explore raw dataset; perform basic structural cleaning                 |  Complete |
| 2  | Clean & normalize complaint text; engineer initial “weak” sentiment labels    |  Complete |
| 3  | Exploratory Data Analysis to validate features and guide modeling             |  Complete |
| 4  | **Tune & Train** an Optimized BiLSTM Model with Keras Tuner                     | Complete |
| 5  | Test & demonstrate the trained model via an **Interactive CLI Application** |  Complete |

---

## 7. Installation

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/deeason7/Consumer-Classification-System.git](https://github.com/deeason7/Consumer-Classification-System.git)
    cd Consumer-Classification-System
    ```

2.  **Create and Activate a Virtual Environment**
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a file named `.env` in the root of the project directory. This file stores the paths to your model and artifacts. Add the following content to it, making sure the paths are correct for your system:
    ```env
    MODEL_PATH="models/sentiment_model_tuned.keras"
    TOKENIZER_PATH="outputs/tokenizer_sentiment.pkl"
    LABEL_ENCODER_PATH="outputs/label_encoder_sentiment.pkl"
    DB_PATH="deployment/predictions.db"
    ```

---

## 8. Usage

The primary way to interact with the model is through the command-line application. The notebooks in the `notebooks/` directory show the full analysis and training workflow.

1.  **Navigate to the project directory** and ensure your virtual environment is activated.

2.  **Run the application:**
    ```sh
    python deployment/app.py
    ```

3.  **Interact with the CLI:**
    The application will load the model and artifacts, then present you with an interactive prompt. You can type or paste a complaint narrative and press **Enter** to get a real-time sentiment prediction.

    **Example Interaction:**

Consumer Complaint Sentiment CLI
Enter the details for a new complaint. Type 'exit' to quit.

Enter complaint narrative: I was charged an unexpected fee for fraud protection services. I have tried contacting customer service multiple times with no response!
Enter product (e.g., 'Mortgage', 'Credit card'): Credit Card
Enter company (e.g., 'Wells Fargo & Company'): Chase
Was the response timely? (yes/no): no

Prediction Result
  Sentiment Label:  extreme_negative
  Confidence Score: 1.00

  Features Used:
    - text_length: 21.0
    - timely_response_binary: 0
    - product_dispute_rate: 0.9679999947547913
    - company_dispute_rate: 1.0192999839782715
    - keyword_flag: 1

 Logged prediction at 2025-08-08T22:19:41.008879


