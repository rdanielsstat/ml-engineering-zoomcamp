from importlib.metadata import version
from fastapi import FastAPI
import uvicorn

print("Fastapi version: " + str(version("fastapi")))
print("Uvicorn version: " + str(uvicorn.__version__))

app = FastAPI(title = "ping")

@app.get("/ping")

def ping():
    return "PONG"

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 9696)