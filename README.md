# Datheon Python & Generative AI Internship

A collection of Python and Generative AI projects developed as part of my internship at Datheon.

## Task 1: Structured LLM Prompt Pipeline & JSON Extractor

### Overview
A Python-based invoice data extraction pipeline that uses structured prompts and Pydantic models to validate invoice information in JSON format.

### Features
- Reusable, parameterized prompt construction
- Few-shot prompting example
- Structured invoice data schema using Pydantic
- Validation of required fields and numeric values
- JSON-formatted output
- Automated tests using pytest
- Environment-variable configuration for the OpenAI API

### Tech Stack
- Python
- OpenAI API
- Pydantic
- python-dotenv
- pytest

### Project Structure
```text
Datheon_Python_GenAI_Internship/
├── task1_json_extractor/
│   ├── main.py
│   ├── schemas.py
│   ├── sample_invoice.txt
│   └── test_extractor.py
├── .env
├── .gitignore
└── requirements.txt
```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yashmishra5013/Datheon-Python-GenAI-Internship.git
   cd Datheon-Python-GenAI-Internship
   ```

2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables in a local `.env` file:
   ```text
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4.1-mini
   ```

   Keep your API key private. Never commit the `.env` file.

### Run the Extractor
```bash
python task1_json_extractor/main.py task1_json_extractor/sample_invoice.txt
```

### Run Tests
```bash
pytest -v
```

The current automated test suite contains five tests covering prompt construction, empty input handling, schema validation, invalid payment status, and negative quantity validation.

### Note
Live invoice extraction requires a working OpenAI API key and available API credits. Automated schema and validation tests can run independently of API credits.

## License
This project was developed for educational and internship purposes.