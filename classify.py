import sys
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No image path provided."}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(json.dumps({"status": "error", "message": f"File not found: {image_path}"}))
        sys.exit(1)

    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, 'flower_cnn_model.h5')
        labels_path = os.path.join(base_dir, 'flower_labels.json')

        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            print(json.dumps({"status": "error", "message": "Model or labels file missing."}))
            sys.exit(1)

        # Load Labels & Model
        with open(labels_path, 'r') as f:
            flower_classes = json.load(f)

        model = tf.keras.models.load_model(model_path)

        # Preprocess Image
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        probabilities = predictions[0]

        top_indices = np.argsort(probabilities)[::-1][:5]

        top_matches = []
        for idx in top_indices:
            name = flower_classes[idx] if idx < len(flower_classes) else f"Species #{idx}"
            confidence = float(probabilities[idx] * 100)
            top_matches.append({
                "species": name.title(),
                "confidence": round(confidence, 2)
            })

        response = {
            "status": "success",
            "primary_prediction": top_matches[0]["species"],
            "confidence": top_matches[0]["confidence"],
            "top_candidates": top_matches
        }

        print(json.dumps(response))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()