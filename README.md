# Image Retrieval using ResNet50

This project implements a retrieval algorithm using ResNet50 from TensorFlow/Keras.  

# Requirements

Before running the script, you need to ensure you have the Tensorflow and opencv libraries installed.

# Folder Structure

The project folder is organized as follows:

project_root/
│
├── main.py
├── image.orig/          # Database images
├── image.query/         # Query images
└── db_embeddings.pkl    # (Generated automatically after first run)

# How to Run

# Run the Main Program

Simply execute the main script:

```bash
python main.py
```

You’ll see a menu:

```
1: Retrieve the most similar image from the database
2: Find similar images with a threshold
3: Relearn Embeddings (After Database Updates)
```

Choose an option by typing the corresponding number.

# Modes Explained

# Option 1: Retrieve Most Similar Image
- Select a query image category in the following list:

'''
1: beach.jpg,
2: mountain.jpg,
3: food.jpg,
4: dinosaur.jpg,
5: flower.jpg,
6: horse.jpg,
7: elephant.jpg
'''

### Option 2: Threshold-Based Similarity
- You enter a similarity threshold (0 to 1).
- Choose a name for the output folder of similar images.
- Select a query image category in the list above.


