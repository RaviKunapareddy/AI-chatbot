# Real-Time Chatbot: Revolutionizing Customer Support with Conversational AI

## Overview

**Note**: This project is for educational purposes, showcasing practical applications of AI and NLP in chatbot development.
This AI Chatbot is a machine learning-powered conversational agent designed to interact with users, classify their queries, and respond effectively based on predefined intents. It utilizes machine learning and natural language processing (NLP) techniques to classify user inputs and generate relevant responses.

This project showcases skills in:
- Natural Language Processing
- Machine Learning
- Python Development

## Features

This project emphasizes:
- Real-world application of natural language understanding techniques.
- Effective implementation of machine learning algorithms for intent classification.
- Robust and modular code architecture for scalability.
- **Customizable Intents**: Train the chatbot with your own intents and responses using the `intents.json` file.
- **Interactive Conversations**: Handles queries such as greetings, programming questions, and more.
- **Scalable**: Easily extend the chatbot by adding new intents and patterns.
- **Pre-Trained Model**: Includes a trained model for quick deployment.

## File Structure

1. **`training.py`**:
   - Prepares and trains the chatbot model using TensorFlow.
   - Processes intents from the `intents.json` file.
   - Saves the trained model and supporting data files.

2. **`chatbot.py`**:
   - Loads the pre-trained model and associated data files.
   - Provides an interactive console for real-time conversations.

3. **`intents.json`**:
   - Defines the chatbot's intents, patterns, and responses.
   - Example:
     ```json
     {
       "tag": "greeting",
       "patterns": ["Hi", "Hello", "Good day"],
       "responses": ["Hello! How can I assist you today?", "Hi there!"]
     }
     ```

4. **`chatbot_model.h5`, `words.pkl`, `classes.pkl`**:
   - The pre-trained model and its associated vocabulary and class data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RaviKunapareddy/AI-chatbot.git
   cd AI-chatbot
   ```

2. (Optional) Modify `intents.json` to customize the chatbot’s responses.

## Usage

### Train the Model
1. To train the chatbot model, run:
   ```bash
   python training.py
   ```
2. The model will be saved as `chatbot_model.h5` along with supporting files.

### Run the Chatbot
1. To start the chatbot, run:
   ```bash
   python chatbot.py
   ```
2. Enter your queries in the console, and the chatbot will respond based on the trained intents.

## Example Interaction
- **Input**: "Hi"
- **Output**: "Hello! How can I assist you today?"

- **Input**: "What is programming?"
- **Output**: "Programming, coding, or software development, means writing computer code to automate tasks."

## Technologies Used
- **Languages**: Python
- **Libraries**: TensorFlow, NLTK

## Future Enhancements
- Integrate a graphical user interface (GUI) for a better user experience.
- Enhance NLP capabilities with transformer-based models like GPT-4.
- Add multi-language support.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or new features.

## License
This project does not currently have a license. It is for educational purposes only, and its use or distribution should be done with the creator's consent.

## Contact
Created by **[Raviteja Kunapareddy](https://www.linkedin.com/in/ravitejak99/)**. Connect for collaboration or questions!

