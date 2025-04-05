#!/usr/bin/env python3
"""
Emergency Services Domain Classifier

This script implements a domain classification component to identify which 
emergency service domain(s) a text belongs to. It provides multi-label 
classification for fire services, police, EMS, etc.
"""
import os
import sys
import re
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('domain_classifier')

# File paths
EMERGENCY_DATASET_PATH = "emergency_services_dataset_emergency_balanced.csv"
DOMAIN_MODEL_PATH = "app/models/domain_classifier_model.pkl"
DOMAIN_VECTORIZER_PATH = "app/models/domain_classifier_vectorizer.pkl"
DOMAIN_BINARIZER_PATH = "app/models/domain_classifier_binarizer.pkl"

# Define emergency service domains
DOMAINS = {
    'fire': [
        'fire', 'firefighter', 'blaze', 'burn', 'flame', 'smoke', 'arson',
        'extinguish', 'hydrant', 'ladder', 'hose', 'firestation', 'firetruck',
        'fire department', 'inferno', 'combustion', 'fire alarm', 'fire chief'
    ],
    'police': [
        'police', 'cop', 'officer', 'patrol', 'sheriff', 'deputy', 'detective',
        'arrest', 'crime', 'criminal', 'law enforcement', 'precinct', 'badge',
        'police department', 'squad car', 'k9', 'swat', 'patrol car', 'handcuff'
    ],
    'ems': [
        'ambulance', 'paramedic', 'emt', 'medic', 'emergency medical',
        'stretcher', 'cpr', 'defibrillator', 'hospital', 'injury', 'wounded',
        'triage', 'first aid', 'trauma', 'medical emergency', 'patient'
    ],
    'coast_guard': [
        'coast guard', 'coastguard', 'lifeguard', 'rescue boat', 'drowning',
        'ocean rescue', 'beach patrol', 'maritime', 'water rescue', 'life jacket',
        'overboard', 'vessel', 'boat rescue', 'swimmer in distress', 'lifesaving'
    ],
    'disaster_response': [
        'disaster', 'emergency', 'flood', 'hurricane', 'tornado', 'earthquake',
        'wildfire', 'landslide', 'evacuation', 'evacuate', 'disaster relief',
        'fema', 'emergency management', 'catastrophe', 'crisis response'
    ]
}

def assign_domains(text):
    """
    Assign emergency service domains to a text based on keyword matching.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of domain labels that apply to the text
    """
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower()
    assigned_domains = []
    
    for domain, keywords in DOMAINS.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                assigned_domains.append(domain)
                break
                
    return assigned_domains if assigned_domains else ['general']

def prepare_training_data(df):
    """
    Prepare training data for the domain classifier.
    
    Args:
        df (pandas.DataFrame): Dataset with 'text' column
        
    Returns:
        tuple: (X, y, mlb) where X is text data, y is binary domain labels
    """
    logger.info("Preparing training data for domain classification")
    
    # Assign domains to each text
    df['domains'] = df['text'].apply(assign_domains)
    
    # Count domain distribution
    domain_counts = {}
    for domains_list in df['domains']:
        for domain in domains_list:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    logger.info(f"Domain distribution: {domain_counts}")
    
    # Convert labels to binary format for multi-label classification
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['domains'])
    
    logger.info(f"Number of domain classes: {len(mlb.classes_)}")
    logger.info(f"Domain classes: {mlb.classes_}")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, mlb

def train_domain_classifier(X_train, y_train):
    """
    Train a multi-label domain classifier.
    
    Args:
        X_train: Training text data
        y_train: Binary domain labels
        
    Returns:
        tuple: (vectorizer, classifier) trained models
    """
    logger.info("Training domain classifier")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    
    # Transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Create and train classifier
    classifier = OneVsRestClassifier(LinearSVC(random_state=42))
    classifier.fit(X_train_vectorized, y_train)
    
    logger.info("Domain classifier training completed")
    
    return vectorizer, classifier

def evaluate_domain_classifier(vectorizer, classifier, X_test, y_test, mlb):
    """
    Evaluate the domain classifier.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        classifier: Trained classifier
        X_test: Test text data
        y_test: True binary domain labels
        mlb: Fitted MultiLabelBinarizer
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating domain classifier")
    
    # Transform test data
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Predict domains
    y_pred = classifier.predict(X_test_vectorized)
    
    # Calculate metrics
    subset_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Subset accuracy: {subset_accuracy:.4f}")
    
    # Calculate per-class metrics
    report = classification_report(
        y_test, y_pred, 
        target_names=mlb.classes_,
        output_dict=True
    )
    
    logger.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=mlb.classes_)}")
    
    # Calculate sample-wise metrics
    correct_samples = 0
    partial_samples = 0
    
    for i in range(len(y_test)):
        true_domains = set(np.where(y_test[i] == 1)[0])
        pred_domains = set(np.where(y_pred[i] == 1)[0])
        
        if true_domains == pred_domains:
            correct_samples += 1
        elif true_domains.intersection(pred_domains):
            partial_samples += 1
    
    exact_match_ratio = correct_samples / len(y_test)
    partial_match_ratio = partial_samples / len(y_test)
    
    logger.info(f"Exact match ratio: {exact_match_ratio:.4f}")
    logger.info(f"Partial match ratio: {partial_match_ratio:.4f}")
    
    return {
        'subset_accuracy': subset_accuracy,
        'report': report,
        'exact_match_ratio': exact_match_ratio,
        'partial_match_ratio': partial_match_ratio
    }

def predict_domains(text, vectorizer, classifier, mlb):
    """
    Predict domains for a given text.
    
    Args:
        text (str): Input text
        vectorizer: Trained TF-IDF vectorizer
        classifier: Trained classifier
        mlb: Fitted MultiLabelBinarizer
        
    Returns:
        list: Predicted domain labels
    """
    # Process input
    if not isinstance(text, str):
        text = str(text)
    
    # Vectorize input
    text_vectorized = vectorizer.transform([text])
    
    # Predict binary domain labels
    domains_binary = classifier.predict(text_vectorized)[0]
    
    # Convert binary labels back to domain names
    domains = mlb.classes_[np.where(domains_binary == 1)[0]].tolist()
    
    # Return general domain if no specific domain is predicted
    return domains if domains else ['general']

def save_models(vectorizer, classifier, mlb):
    """
    Save the trained models to disk.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        classifier: Trained classifier
        mlb: Fitted MultiLabelBinarizer
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(DOMAIN_MODEL_PATH), exist_ok=True)
    
    # Save models
    joblib.dump(vectorizer, DOMAIN_VECTORIZER_PATH)
    joblib.dump(classifier, DOMAIN_MODEL_PATH)
    joblib.dump(mlb, DOMAIN_BINARIZER_PATH)
    
    logger.info(f"Domain classifier saved to {DOMAIN_MODEL_PATH}")
    logger.info(f"Domain vectorizer saved to {DOMAIN_VECTORIZER_PATH}")
    logger.info(f"Domain binarizer saved to {DOMAIN_BINARIZER_PATH}")

def load_models():
    """
    Load trained domain classification models.
    
    Returns:
        tuple: (vectorizer, classifier, mlb) trained models
    """
    try:
        vectorizer = joblib.load(DOMAIN_VECTORIZER_PATH)
        classifier = joblib.load(DOMAIN_MODEL_PATH)
        mlb = joblib.load(DOMAIN_BINARIZER_PATH)
        logger.info("Successfully loaded domain classification models")
        return vectorizer, classifier, mlb
    except Exception as e:
        logger.error(f"Error loading domain classification models: {str(e)}")
        return None, None, None

def main():
    """Main function to train and evaluate the domain classifier."""
    try:
        logger.info("Starting domain classifier training and evaluation")
        
        # Load emergency services dataset
        logger.info(f"Loading dataset from {EMERGENCY_DATASET_PATH}")
        df = pd.read_csv(EMERGENCY_DATASET_PATH)
        logger.info(f"Loaded {len(df)} samples")
        
        # Prepare training data
        X_train, X_test, y_train, y_test, mlb = prepare_training_data(df)
        
        # Train domain classifier
        vectorizer, classifier = train_domain_classifier(X_train, y_train)
        
        # Evaluate domain classifier
        metrics = evaluate_domain_classifier(vectorizer, classifier, X_test, y_test, mlb)
        
        # Save models
        save_models(vectorizer, classifier, mlb)
        
        # Example prediction
        example_text = "Firefighters responded to a three-alarm fire at an apartment building."
        domains = predict_domains(example_text, vectorizer, classifier, mlb)
        logger.info(f"Example text: '{example_text}'")
        logger.info(f"Predicted domains: {domains}")
        
        logger.info("Domain classifier training and evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in domain classifier: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 