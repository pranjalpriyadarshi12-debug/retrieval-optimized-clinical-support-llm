import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
PROJECT_ID = os.getenv("PROJECT_ID")
API_VERSION = os.getenv("API_VERSION")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
