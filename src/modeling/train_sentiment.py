# src/modeling/train_sentiment.py
"""
Defines the end-to-end pipeline for training and evaluating the sentiment classification model.
It includes data preparation, model building, hyperparameter tuning, cross-validation, and artifact saving.
"""
import pickle
import os
import sys
import json
import tensorflow as tf
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import keras_tuner as kt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

DATA_PATH = os.path.join(project_root, "data/processed/consumer_complaints_final.csv")

# Module-level Constants
VOCAB_SIZE = 25000
MAX_LEN = 250

def prepare_data(path: str, vocab_size: int, max_len: int) -> Tuple:
    """Loads, preprocesses, and splits the data for sentiment classification."""
    df = pd.read_csv(path)
    df = df[df['text_cleaned'].notna()].copy()

    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])

    X_text = df['text_cleaned']
    X_struct = df[['text_length', 'timely_response_binary', 'product_dispute_rate',
                   'company_dispute_rate', 'keyword_flag', 'sentiment_intensity']].values.astype(np.float32)
    y = df['sentiment_encoded'].values

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_text)
    X_seq = pad_sequences(tokenizer.texts_to_sequences(X_text), maxlen=max_len)

    X_seq_train, X_seq_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
        X_seq, X_struct, y, test_size=0.2, stratify=y, random_state=42
    )

    return (X_seq_train, X_struct_train, y_train,
            X_seq_test, X_struct_test, y_test,
            tokenizer, le)


def build_model(hp: kt.HyperParameters, vocab_size: int, max_len: int) -> Model:
    """Constructs a hybrid Keras model, adaptable for Keras Tuner."""
    # Define search space for hyperparameters
    lstm_units = hp.Int('lstm_units', 64, 256, step=64)
    dropout_rate = hp.Float('dropout_rate', 0.2, 0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    text_input = Input(shape=(max_len,), name="text_input")
    struct_input = Input(shape=(6,), name="struct_input")

    x = Embedding(input_dim=vocab_size, output_dim=128)(text_input)
    x = Bidirectional(LSTM(units=lstm_units))(x)
    x = Dropout(rate=dropout_rate)(x)

    merged = Concatenate()([x, struct_input])
    merged = Dense(units=lstm_units, activation='relu')(merged)
    merged = Dropout(rate=dropout_rate)(merged)
    output = Dense(3, activation='softmax')(merged)

    model = Model(inputs=[text_input, struct_input], outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


def run_hyperparameter_tuning(X_seq_train, X_struct_train, y_train, models_dir: str) -> Dict[str, Any]:
    """Performs hyperparameter tuning using Keras Tuner."""
    model_builder = lambda hp: build_model(hp, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory=os.path.dirname(models_dir),
        project_name=os.path.basename(models_dir)
    )

    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    X_seq_t, X_seq_v, X_struct_t, X_struct_v, y_t, y_v = train_test_split(
        X_seq_train, X_struct_train, y_train, test_size=0.2, random_state=42
    )

    tuner.search(
        {'text_input': X_seq_t, 'struct_input': X_struct_t}, y_t,
        epochs=50,
        validation_data=({'text_input': X_seq_v, 'struct_input': X_struct_v}, y_v),
        callbacks=[stop_early]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    Best hyperparameters found:
    - LSTM Units: {best_hps.get('lstm_units')}
    - Dropout Rate: {best_hps.get('dropout_rate')}
    - Learning Rate: {best_hps.get('learning_rate')}
    """)
    return best_hps.values


def train_with_cross_validation(X_seq, X_struct, y, best_hps_values: Dict[str, Any], n_splits: int = 5) -> Tuple[
    Model, Dict[str, List[float]]]:
    """Trains the model using stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y)):
        print(f"\n----------- FOLD {fold + 1}/{n_splits} -------")
        hp = kt.HyperParameters()
        hp.values = best_hps_values
        model = build_model(hp, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)

        X_seq_t, X_struct_t, y_t = X_seq[train_idx], X_struct[train_idx], y[train_idx]
        X_seq_v, X_struct_v, y_v = X_seq[val_idx], X_struct[val_idx], y[val_idx]

        class_weights = dict(
            zip(np.unique(y_t), compute_class_weight(class_weight='balanced', classes=np.unique(y_t), y=y_t)))

        history = model.fit(
            {'text_input': X_seq_t, 'struct_input': X_struct_t}, y_t,
            validation_data=({'text_input': X_seq_v, 'struct_input': X_struct_v}, y_v),
            epochs=30, batch_size=64, class_weight=class_weights,
            callbacks=[EarlyStopping(patience=4), ReduceLROnPlateau(patience=2)],
            verbose=2
        )

        best_epoch_idx = np.argmin(history.history['val_loss'])
        for key in cv_history:
            cv_history[key].append(history.history[key][best_epoch_idx])

    print("\n--- CROSS-VALIDATION SUMMARY -------")
    for key in cv_history:
        print(f"Average {key}: {np.mean(cv_history[key]):.4f} (+/- {np.std(cv_history[key]):.4f})")

    print("\n------- TRAINING FINAL MODEL ON ALL DATA-----")
    final_hp = kt.HyperParameters()
    final_hp.values = best_hps_values
    final_model = build_model(final_hp, vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    class_weights = dict(zip(np.unique(y), compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)))

    final_model.fit(
        {'text_input': X_seq, 'struct_input': X_struct}, y,
        epochs=15,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[ReduceLROnPlateau(monitor='loss', patience=2)],
        verbose=2
    )

    return final_model, cv_history


def train_and_evaluate_sentiment(
        tune: bool = False,
        data_path: str = DATA_PATH,
        models_dir: str = "models",
        outputs_dir: str = "outputs"
):
    """Main orchestrator for the full sentiment training pipeline."""
    model_path = os.path.join(models_dir, "sentiment_model.keras")
    tokenizer_path = os.path.join(outputs_dir, "tokenizer_sentiment.pkl")
    encoder_path = os.path.join(outputs_dir, "label_encoder_sentiment.pkl")
    report_path = os.path.join(outputs_dir, "sentiment_classification_report.json")
    cv_scores_path = os.path.join(outputs_dir, "cross_validation_scores.csv")

    (X_seq_train, X_struct_train, y_train,
     X_seq_test, X_struct_test, y_test,
     tokenizer, le) = prepare_data(data_path, VOCAB_SIZE, MAX_LEN)

    if tune:
        tuning_dir = os.path.join(models_dir, 'sentiment_tuning')
        best_hps_values = run_hyperparameter_tuning(X_seq_train, X_struct_train, y_train, tuning_dir)
    else:
        best_hps_values = {'lstm_units': 128, 'dropout_rate': 0.3, 'learning_rate': 0.001}
        print("Using default hyperparameters. To tune, call with tune=True.")

    final_model, cv_results = train_with_cross_validation(
        X_seq_train, X_struct_train, y_train, best_hps_values
    )

    y_pred_prob = final_model.predict({"text_input": X_seq_test, "struct_input": X_struct_test})
    y_pred = np.argmax(y_pred_prob, axis=1)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    final_model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(cv_scores_path, index=False)

    print(f"\nSentiment model training complete. Artifacts saved to {models_dir} and {outputs_dir}")


if __name__ == '__main__':
    train_and_evaluate_sentiment(tune=False)