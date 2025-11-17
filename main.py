import os
import cv2 as cv
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from glob import glob
import pickle, shutil
from pathlib import Path

# Normalize path separators consistently
sep = os.sep

### Annotated Files ###
files = []

folder_path = f"image.orig".replace("/", sep).replace("\\", sep)

for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if item.lower().endswith(".jpg"):
        files.append(item)

files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

print(len(files))
labels = [9, 1, 9, 9, 4, 7, 5, 6, 2, 3]

img_labels = []
for i in labels:
    img_labels += [i] * 100

filename_label = dict(zip(files, img_labels))

### Preparing retrieval from saved embeddings

IMG_WIDTH, IMG_HEIGHT = 224, 224
EMBEDDINGS_FILE = f"db_embeddings.pkl"

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

### Extract embeddings
def extract_resnet_features(img_path, model):
    img = cv.imread(img_path)
    img_resized = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_array = img_resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.flatten()
    features = features / np.linalg.norm(features)
    return features

### Build embeddings
def build_database_embeddings(database_dir="image.orig"):
    database_dir = database_dir.replace("/", sep).replace("\\", sep)
    database_files = sorted(glob(os.path.join(database_dir, "*.jpg")))
    db_features_list = []

    for file in database_files:
        features = extract_resnet_features(file, model)
        db_features_list.append(features)

    db_features = np.array(db_features_list)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"features": db_features, "paths": database_files}, f)
    print(f"Saved {len(database_files)} embeddings to {EMBEDDINGS_FILE}")

### Retrieval
def retrieval():
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings not found! Run build_database_embeddings() first.")
        return

    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    db_features = data["features"]
    database_files = data["paths"]

    print("1: beach\n2: mountain\n3: food\n4: dinosaur\n5: flower\n6: horse\n7: elephant")
    choice = input("Type the number to choose a category: ")

    query_file_map = {
        '1': f"image.query{sep}beach.jpg",
        '2': f"image.query{sep}mountain.jpg",
        '3': f"image.query{sep}food.jpg",
        '4': f"image.query{sep}dinosaur.jpg",
        '5': f"image.query{sep}flower.jpg",
        '6': f"image.query{sep}horse.jpg",
        '7': f"image.query{sep}elephant.jpg"
    }

    if choice not in query_file_map:
        print("Invalid choice")
        return

    src_path = query_file_map[choice]
    query_features = extract_resnet_features(src_path, model)
    print(f"You chose: {os.path.basename(src_path)}")

    cv.imshow("Query", cv.imread(src_path))

    distances = np.linalg.norm(db_features - query_features, axis=1)
    closest_idx = np.argmin(distances)
    closest_file = database_files[closest_idx]

    print(f"Most similar image: {closest_file} dist={distances[closest_idx]:.4f}")

    closest_img = cv.imread(closest_file)
    cv.imshow("Closest Match", closest_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit()

### Threshold similarity
def threshold_similarity(model, threshold, output_folder):
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings not found! Run build_database_embeddings() first.")
        return

    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    db_features = data["features"]
    database_files = data["paths"]

    print("1: beach\n2: mountain\n3: food\n4: dinosaur\n5: flower\n6: horse\n7: elephant")
    choice = input("Type the number to choose a category: ")
    query_label = int(choice)

    query_file_map = {
        '1': f"image.query{sep}beach.jpg",
        '2': f"image.query{sep}mountain.jpg",
        '3': f"image.query{sep}food.jpg",
        '4': f"image.query{sep}dinosaur.jpg",
        '5': f"image.query{sep}flower.jpg",
        '6': f"image.query{sep}horse.jpg",
        '7': f"image.query{sep}elephant.jpg"
    }

    if choice not in query_file_map:
        print("Invalid choice")
        return

    src_path = query_file_map[choice]
    query_features = extract_resnet_features(src_path, model)
    print(f"You chose: {os.path.basename(src_path)}")

    distances = np.linalg.norm(db_features - query_features, axis=1)
    similarity = 1 / (1 + distances)
    similarity_rescaled = (similarity - min(similarity)) / (max(similarity) - min(similarity))

    images = [database_files[i] for i in range(len(similarity)) if similarity_rescaled[i] > threshold]

    def folder(src, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
        for image_path in src:
            if os.path.exists(image_path):
                file_name = os.path.basename(image_path)
                shutil.copy2(image_path, os.path.join(folder, file_name))

    folder(images, output_folder)
    print(f"Found {len(images)} images above threshold {threshold} -> {output_folder}")

    true_class_files = [f for f, lbl in filename_label.items() if lbl == query_label]
    TP = sum(1 for f in images if filename_label.get(os.path.basename(f), -1) == query_label)
    FP = len(images) - TP
    FN = len(true_class_files) - TP

    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f"True Positives: {TP}")
    print(f"Precision: {precision_val:.3f}")
    print(f"Recall: {recall_val:.3f}")


def main():
    print("1: Retrieve most similar image")
    print("2: Find similar images with threshold")
    print("3: Relearn embeddings")
    number = int(input("Choose: "))

    if number == 1:
        retrieval()
    elif number == 2:
        threshold = float(input("Threshold (0-1): "))
        if not 0 <= threshold <= 1:
            print("Invalid threshold!")
            return
        output = input("Output folder: ")
        threshold_similarity(model, threshold, output)
    elif number == 3:
        build_database_embeddings("image.orig")
    else:
        print("Invalid input!")
        exit()

main()
