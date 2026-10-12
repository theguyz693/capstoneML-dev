"""
research_server.py
Starts the EXISTING server (server.py, unchanged) plus the research engine endpoints and dashboard.

    python research_server.py
    -> http://localhost:8000/            original 3D dashboard
    -> http://localhost:8000/research    research dashboard (method race, scoreboard, kiosk signs, weak spots)
"""
import os

from fastapi.responses import FileResponse

from server import app                      # existing app, unchanged
from evac.api import router

app.include_router(router)
HERE = os.path.dirname(os.path.abspath(__file__))


@app.get("/research")
def research_page():
    return FileResponse(os.path.join(HERE, "research.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("research_server:app", host="0.0.0.0", port=8000, reload=False)
