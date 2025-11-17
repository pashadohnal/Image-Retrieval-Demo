# Image Retrieval using ResNet50

This project implements a retrieval algorithm using ResNet50 from TensorFlow/Keras.  

# Requirements

Before running the script, you need to ensure you have the Tensorflow and opencv libraries installed.

# Folder Structure

The project folder is organized as follows:

```
project_root/
│
├── main.py
├── image.orig/             # Database images
├── image.query/            # Query images
├── evaluation_results.csv  # Table of the results depending on the threshold
└── db_embeddings.pkl       # (Generated automatically after first run)
```

# How to Run

### Run the Main Program

Simply execute the main script:

```bash
python main.py
```

You’ll see a menu:

```
1: Retrieve the most similar image
2: Find similar images with threshold
3: Relearn Embeddings
4: Evaluate performance
```

Choose an option by typing the corresponding number.

# Modes Explained

### Option 1: Retrieve Most Similar Image
- Select a query image category in the following list:

```
1: beach.jpg,
2: mountain.jpg,
3: food.jpg,
4: dinosaur.jpg,
5: flower.jpg,
6: horse.jpg,
7: elephant.jpg
```

### Option 2: Threshold-Based Similarity
- You enter a similarity threshold (0 to 1).
- Choose a name for the output folder of similar images.
- Select a query image category in the list above.

### Option 3: Relearn Embendings
Use ResNet50 to recreate the embendings, and rewrite the embendings file.

### Option 4: Evaluate performance
Creats a csv file comparing the recall and precision depending on different category and threshold.


