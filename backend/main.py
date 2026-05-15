from fastapi import FastAPI
from database import Base , engine
from fastapi.middleware.cors import CORSMiddleware
from routers import csp_router, hc_router, sa_router, greedy_router, schedules ,team_router

Base.metadata.create_all(bind=engine)
app = FastAPI(title="League Scheduler API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, safe for local demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register specific algorithms
app.include_router(csp_router.router)
app.include_router(hc_router.router)
app.include_router(sa_router.router)
app.include_router(greedy_router.router)
app.include_router(team_router.router)

# Register management (CRUD)
app.include_router(schedules.router)

@app.get("/")
def root():
    return {
        "status": "Online",
        "message": "AI Scheduler Backend is ready. Visit /docs for Swagger UI."
    }