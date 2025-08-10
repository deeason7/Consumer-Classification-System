# Consumer Complaint Sentiment Insights and Classification System

An end-to-end NLP pipeline that classifies the sentiment of consumer financial complaints using a hybrid deep learning model, accessible via an interactive web application and a command-line interface.

---

## 1. Context & Motivation

The **Consumer Financial Protection Bureau (CFPB)** collects thousands of consumer complaints across financial services. Understanding the emotional intensity of these complaint narratives (Neutral, Negative, Extreme Negative) enables us to:
* Detect systemic issues in financial services.
* Improve customer-service strategies at institutions.
* Inform policy decisions and enforcement priorities.
* Prioritize high-risk cases for investigation.

---

## 2. Problem Statement

How can we leverage NLP and deep learning to build a scalable, production-ready pipeline that classifies consumer-complaint narratives into three sentiment levels and provides a workflow for managing them:
* Neutral
* Negative
* Extreme Negative

---

## 3. Key Features

* **Hybrid Deep Learning Model**: The system's core is a BiLSTM model that processes both the complaint text and structured metadata (like text length and keyword flags) for more nuanced predictions.
* **Hyperparameter Tuning**: Utilizes **Keras Tuner** to systematically find the optimal model architecture and parameters, significantly boosting performance.
* **Dual-Interface Web Application**: A user-friendly **Flask** web app provides two distinct views:
    * **Customer Portal**: Allows users to submit new complaints and check the status of existing ones using a unique reference ID.
    * **Agent Dashboard**: An internal tool that displays all complaints, automatically prioritized by a sentiment-based score, allowing for efficient review and management.
* **Interactive CLI**: For developers and power users, an interactive Command-Line Interface (`app.py`) offers an alternative way to get real-time predictions.
* **Persistent Complaint Tracking**: Submissions are automatically timestamped and saved to a local **SQLite** database, creating a verifiable log and tracking the status of each complaint from "Submitted" to "Responded."
* **Modular & Scalable Design**: The code is cleanly separated into modules for data processing and a dedicated `deployment/` package for inference.

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
| **Deployment & Storage** | Flask Web Application, CLI, SQLite, Dotenv                 |
| **Version Control** | Git & GitHub                                               |

---

## 6. Project Workflow & Status

| #  | Objective                                                                     | Status    |
| :--| :---------------------------------------------------------------------------- | :-------- |
| 1  | Load & explore raw dataset; perform basic structural cleaning                 | Complete |
| 2  | Clean & normalize complaint text; engineer initial “weak” sentiment labels    |  Complete |
| 3  | Exploratory Data Analysis to validate features and guide modeling             |  Complete |
| 4  | **Tune & Train** an Optimized BiLSTM Model with Keras Tuner                     |  Complete |
| 5  | **Deploy** the model via an **Interactive Web App & CLI** |  Complete |

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

The application can be run as a Web Application (recommended) or a Command-Line Interface.

1.  **Navigate to the project directory** and ensure your virtual environment is activated.

2.  **Run the main application launcher:**
    ```sh
    python deployment/app.py
    ```

3.  **Choose an interface from the menu:**
    * **Option 1: Web Interface (Recommended)**
        * Select `1` to launch the Flask web server.
        * Open your browser and navigate to **`http://127.0.0.1:5001`**.
        * From the web page, you can submit new complaints, check the status of a past submission, or navigate to the agent dashboard.
        * **Agent Dashboard URL**: **`http://127.0.0.1:5001/dashboard`**

    * **Option 2: Command-Line Interface (CLI)**
        * Select `2` to launch the interactive CLI in your terminal.
        * Follow the prompts to enter a complaint narrative and its details to get a real-time prediction.
