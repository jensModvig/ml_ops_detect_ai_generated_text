from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI()

# Define a root `/` endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}

