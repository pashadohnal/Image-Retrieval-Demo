import os
import cv2 as cv
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from glob import glob
import pickle

IMG_WIDTH, IMG_HEIGHT = 224, 224  # ResNet50 default input size
EMBEDDINGS_FILE = "db_embeddings.pkl"

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_resnet_features(img_path, model):
    img = cv.imread(img_path)
    img_resized = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_array = img_resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # preprocess for ResNet50
    features = model.predict(img_array)
    features = features.flatten()
    # Normalize to unit vector (optional, helps with cosine similarity)
    features = features / np.linalg.norm(features)
    return features

def retrieval():
    # --- Load precomputed embeddings ---
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings not found! Run build_database_embeddings() first.")
        return

    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    db_features = data["features"]
    database_files = data["paths"]

    # --- Choose query image ---
    print("1: beach\n2: mountain\n3: food\n4: dinosaur\n5: flower\n6: horse\n7: elephant")
    choice = input("Type the number to choose a category: ")

    query_file_map = {
        '1': 'beach.jpg',
        '2': 'mountain.jpg',
        '3': 'food.jpg',
        '4': 'dinosaur.jpg',
        '5': 'flower.jpg',
        '6': 'horse.jpg',
        '7': 'elephant.jpg'
    }

    if choice not in query_file_map:
        print("Invalid choice")
        return

    src_path = os.path.join("image.query", query_file_map[choice])
    query_features = extract_resnet_features(src_path, model)
    print(f"You chose: {query_file_map[choice]}")

    cv.imshow("Query", cv.imread(src_path))

    # --- Compute distances ---
    distances = np.linalg.norm(db_features - query_features, axis=1)
    closest_idx = np.argmin(distances)
    closest_file = database_files[closest_idx]

    print(f"The most similar image is {closest_file} with distance {distances[closest_idx]:.4f}")

    closest_img = cv.imread(closest_file)
    cv.imshow("Closest Match", closest_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    exit
    
def threshold_similarity(threshold):
    threshold
    pass

def precision():
    pass

def recall():
    pass

def relearn_embeddings(path, model):
    pass





