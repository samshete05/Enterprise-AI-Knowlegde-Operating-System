import os
import uvicorn

if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host=host, port=port)