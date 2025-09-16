@echo off
echo ========================================
echo   AI Research Agent - Backend FastAPI
echo ========================================
echo.

echo Activating conda environment...
call conda activate ai_ra
if %errorlevel% neq 0 (
    echo ERROR: Could not activate conda environment 'ai_ra'
    echo Please create the environment first:
    echo conda create -n ai_ra python=3.10 -y
    echo conda activate ai_ra
    echo pip install -r deployment/requirements.txt
    pause
    exit /b 1
)

echo.
echo Starting FastAPI backend...
echo API will be available at: http://localhost:8000
echo API docs will be available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn ai_research_agent.src.api.main:app --reload --port 8000

