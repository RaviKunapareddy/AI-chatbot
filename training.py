import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize the patterns and lemmatize each word
for intent in intents['intents']:
    if not intent['patterns']:
        print(f"Skipping intent with empty patterns: {intent['tag']}")
        continue
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Tokenize the pattern into words
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and ignore specified letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# Save the processed words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare the training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]

    # Create a bag of words: 1 if the word exists in the pattern, otherwise 0
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    # Create an output row: 1 for the corresponding class, 0 for others
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Check for inconsistent shapes in training data
for idx, (bag, output_row) in enumerate(training):
    if len(bag) != len(words) or len(output_row) != len(classes):
        print(f"Inconsistent shape at index {idx}: bag length = {len(bag)}, output_row length = {len(output_row)}")

# Shuffle the training data
random.shuffle(training)

# Convert to numpy arrays
try:
    train_x = np.array([np.array(sample[0]) for sample in training])
    train_y = np.array([np.array(sample[1]) for sample in training])
except ValueError as e:
    print(f"Error converting training data to numpy arrays: {e}")
    print("Check the data for inconsistencies.")
    exit()

# Debugging: Print the shapes of the final arrays
print(f"Shape of train_x: {train_x.shape}")
print(f"Shape of train_y: {train_y.shape}")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and Save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)

# Save the trained model
model.save('chatbot_model.h5',hist)

print("Model training complete and saved!")
