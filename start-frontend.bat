@echo off
echo ========================================
echo   AI Research Agent - Frontend React
echo ========================================
echo.

echo Verifying Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js version:
node --version

echo.
echo Checking if dependencies are installed...
if not exist "frontend-react\node_modules" (
    echo Installing dependencies...
    cd frontend-react
    npm install
    cd ..
) else (
    echo Dependencies already installed.
)

echo.
echo Starting React frontend...
echo Frontend will be available at: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

cd frontend-react
npm run dev

