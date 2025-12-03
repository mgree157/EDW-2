Terminal 1 – Backend (FastAPI / Uvicorn)

In root (EDW-2)
source venv/bin/activate
# Load all vars from .env into the shell (including GROQ_API_KEY)
export $(cat .env | xargs)

# Sanity check that the env var is now visible
python -c "import os; print('GROQ_API_KEY =', os.getenv('GROQ_API_KEY'))"

# Start backend
uvicorn PersonA.Backend:app --reload