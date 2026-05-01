#!/usr/bin/env python3
"""
Inference script for bot detection.
Takes an interactions JSON file and predicts whether it's human or bot.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add script directory to path to ensure imports work
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import feature extraction functions from extract_features
from botbuster.constants import LABEL_HUMAN, METADATA_COLUMNS
from extract_features import process_json_file


def load_model(model_path):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    return model


def get_feature_columns_from_csv(csv_path):
    """Get feature column names from the features CSV (excluding metadata).
    Preserves the original column order from the CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Features CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path, nrows=1)  # Just read header
    exclude_cols = list(METADATA_COLUMNS)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols  # Preserve original order, don't sort


def prepare_features_for_prediction(features_dict, feature_columns):
    """Convert features dictionary to a DataFrame with correct column order."""
    # Create a single row with values in the correct column order
    values = [features_dict.get(col, 0) for col in feature_columns]
    feature_row = pd.DataFrame([values], columns=feature_columns)
    
    # Fill NaN and inf values
    feature_row = feature_row.fillna(0).replace([np.inf, -np.inf], 0)
    
    return feature_row


def predict_single_file(model, json_file_path, feature_columns):
    """Predict on a single JSON file."""
    print(f"\n{'='*60}")
    print(f"Processing: {json_file_path}")
    print(f"{'='*60}")
    
    # Extract features from the JSON file
    all_features = process_json_file(json_file_path, label=None)
    
    if len(all_features) == 0:
        print("No sessions found in the file!")
        return
    
    print(f"\nFound {len(all_features)} session(s) in the file.")
    
    results = []
    
    for i, features_dict in enumerate(all_features):
        session_id = features_dict.get('session_id', f'session_{i}')
        user_id = features_dict.get('user_id', 'unknown')
        
        # Prepare features for prediction
        feature_row = prepare_features_for_prediction(features_dict, feature_columns)
        
        # Make prediction
        prediction = model.predict(feature_row)[0]
        probabilities = model.predict_proba(feature_row)[0]
        
        # Get confidence
        confidence = probabilities[prediction] * 100
        
        result = {
            'user_id': user_id,
            'session_id': session_id,
            'prediction': 'Human' if prediction == LABEL_HUMAN else 'Bot',
            'prediction_label': int(prediction),
            'confidence': confidence,
            'human_probability': probabilities[1] * 100,
            'bot_probability': probabilities[0] * 100
        }
        results.append(result)
        
        # Print result for this session
        print(f"\n--- Session {i+1}: {session_id} (User: {user_id}) ---")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"  Human probability: {result['human_probability']:.2f}%")
        print(f"  Bot probability: {result['bot_probability']:.2f}%")
    
    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Bot detection inference script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py interactions.json
  python inference.py --model models/v0/bot_detection_model.pkl interactions.json
  python inference.py --model models/v0/bot_detection_model.pkl --csv features.csv interactions.json
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the interactions JSON file to analyze'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/v0/bot_detection_model.pkl',
        help='Path to the trained model file (default: models/v0/bot_detection_model.pkl)'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default='features.csv',
        help='Path to features CSV file to get feature column order (default: features.csv)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_file = Path(args.input_file)
    model_path = script_dir / args.model if not Path(args.model).is_absolute() else Path(args.model)
    csv_path = script_dir / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        # Load model
        model = load_model(model_path)
        
        # Get feature columns
        print(f"\nLoading feature column order from {csv_path}...")
        feature_columns = get_feature_columns_from_csv(csv_path)
        print(f"Found {len(feature_columns)} features")
        
        # Make predictions
        results = predict_single_file(model, input_file, feature_columns)
        
        if results:
            # Print summary
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            human_count = sum(1 for r in results if r['prediction'] == 'Human')
            bot_count = sum(1 for r in results if r['prediction'] == 'Bot')
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"Total sessions: {len(results)}")
            print(f"  Human: {human_count} ({human_count/len(results)*100:.1f}%)")
            print(f"  Bot: {bot_count} ({bot_count/len(results)*100:.1f}%)")
            print(f"Average confidence: {avg_confidence:.2f}%")
            
            # Overall prediction (majority vote)
            if human_count > bot_count:
                overall = "Human"
            elif bot_count > human_count:
                overall = "Bot"
            else:
                overall = "Tie"
            
            print(f"\nOverall prediction: {overall}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

