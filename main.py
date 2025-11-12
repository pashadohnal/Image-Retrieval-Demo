import os
import cv2 as cv
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from glob import glob
import pickle, shutil
from pathlib import Path

IMG_WIDTH, IMG_HEIGHT = 224, 224  # ResNet50 default input size
EMBEDDINGS_FILE = "db_embeddings.pkl"

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#Extracting embeddings from the images in the folder
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

#Saving embeddings into a file
def build_database_embeddings(database_dir="image.orig"):
    database_files = sorted(glob(os.path.join(database_dir, "*.jpg")))
    db_features_list = []

    for file in database_files:
        features = extract_resnet_features(file, model)
        db_features_list.append(features)

    db_features = np.array(db_features_list)
    # Save embeddings and file paths
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"features": db_features, "paths": database_files}, f)
    print(f"Saved {len(database_files)} embeddings to {EMBEDDINGS_FILE}")

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

    exit()
    
def threshold_similarity(model, threshold, output_folder):

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
    choice = input("Type the number to choose a category : ")

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

    cv.imshow("Query", cv.resize(cv.imread(src_path), (256, 256)))

    # --- Compute distances ---
    distances = np.linalg.norm(db_features - query_features, axis=1)

    def distance_to_similarity(distances):
        return 1 / (1 + distances)
    
    similarity = distance_to_similarity(distances)
    images = []
    for i in range(len(similarity)):
        if similarity[i] > threshold :
            images.append(database_files[i])

    def folder(src, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
    
        for image_path in src:
            if os.path.exists(image_path):

                file_name = os.path.basename(image_path)
                path = os.path.join(folder, file_name)
            
                shutil.copy2(image_path, path)

    folder(images, output_folder)
    print(f"Found {len(images)} images with similarity above {threshold}. Saved to folder '{output_folder}'.")
    exit

def precision():
    pass

def recall():
    pass

def relearn_embeddings(path, model):
    pass



def main():
    print("1: Retrieve the most similar image from the database")
    print("2: Find similar images with a threshold")
    print("3: Relearn Embeddings (After Database Updates)")
    number = int(input("Type in the number to choose a function and type enter to confirm\n"))
    if number == 1:
        retrieval()
    elif number == 2:
        threshold = float(input("Choose the threshold (0-1) : "))
        if threshold > 1 or threshold < 0:
            print("Invalid threshold !")
            return
        output = input("Choose the folder name for similar images : ")
        threshold_similarity(model, threshold, output)
    else:
        print("Invalid input !")
        exit()

main()