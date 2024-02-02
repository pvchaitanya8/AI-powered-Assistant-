# Chatbot Fusion Project

This project combines two chatbot functionalities: CSV Chatbot and TXT Chatbot. The CSV Chatbot analyzes data from CSV files, while the TXT Chatbot processes information from text files.

## Requirements

Ensure you have the required packages installed before running the project:

```bash
%pip install langchain openai tabulate llama-index
```

## Setup

1. Obtain OpenAI API Key: Replace the placeholder `*********************************` in the code with your actual OpenAI API key.

2. Prepare CSV Data: Ensure you have a CSV file named `data.csv` in the `knowledge` folder for the CSV Chatbot.

3. Prepare TXT Data: Place your text documents in the `Knowledge` folder for the TXT Chatbot.

## Running the Project

Execute the combined project script:

```bash
python combined_chatbot.py
```

### User Options

1. **Ask Question (Option 1):** Enter a question, and the combined chatbot will provide responses from both CSV and TXT chatbots.

2. **Stop (Option 0):** Terminate the program.

## CSV Chatbot Functionality

- Analyzes questions based on data stored in a CSV file (`knowledge/data.csv`).
- Updates CSV data in SQLite database after each interaction.

## TXT Chatbot Functionality

- Processes questions using a GPT-based model (OpenAI's text-ada-001) on text documents in the `Knowledge` folder.
- Utilizes a GPTVectorStoreIndex for efficient query handling.

## Note

- Adjust file paths, OpenAI API key, and other parameters as needed.
- Ensure proper data files and folders are in place before running.
