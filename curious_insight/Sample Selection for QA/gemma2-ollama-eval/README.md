# Gemma 2 Ollama Evaluation

This project is designed to evaluate the Gemma 2 27B model using the Ollama API. It allows users to send questions to the model and receive answers, which are then saved to a CSV file for further analysis.

## Project Structure

```
gemma2-ollama-eval
├── src
│   └── evaluate_gemma.py       # Main script for evaluating the Gemma model
├── scripts
│   └── pull_gemma_model.sh      # Script to pull the Gemma 2 model from Ollama
├── requirements.txt             # Python dependencies
├── .env.example                  # Template for environment variables
├── .vscode
│   └── launch.json              # Debugging configuration for VS Code
├── README.md                    # Project documentation
└── LICENSE                      # Licensing information
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd gemma2-ollama-eval
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Copy the `.env.example` file to `.env` and update the necessary configurations, such as API endpoints.

4. **Pull the Gemma 2 model:**
   Run the following command to pull the model from Ollama:
   ```
   bash scripts/pull_gemma_model.sh
   ```

## Usage

To evaluate questions using the Gemma 2 model, run the main script:
```
python src/evaluate_gemma.py
```

Make sure to provide the input CSV file with questions and contexts as specified in the script.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.