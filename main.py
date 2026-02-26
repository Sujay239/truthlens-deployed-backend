from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize database connection and other settings here
    pass

@app.get("/")
def read_root():
    return {"Hello": "World"}