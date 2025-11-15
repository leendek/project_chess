# Project advanced AI

## Object detection

Train an object detection model capable of detecting the current board setup for a chess.com game. Select a model and train it using relevant chess images. The model should be able to detect all the pieces together with their color in various settings. Test the model on images that you made yourself and demonstrate that it works correctly. Check how good it performs on other images, not from chess.com. Don’t forget to do data analysis on your dataset.

### existing models
I found a few models on chess.com images on [roboflow](https://universe.roboflow.com). 
I selected the model with the highest precision and recall: https://universe.roboflow.com/grupo-12/grupo12-fzbt4 . I couldn't find a way to use this model locally. It was possible to download the dataset in COCO format.
Maybe look into the huggingface model a bit more?
### training
Similar to the training on chickens, didn't change any parameters, no image augmentation etc

### test on images that you made yourself
in data/browserscreenshots there are some screenshots I made myself on chess.com. I added some code to extract only the chessboard in the notebook, output was stored in data/chessboards.
