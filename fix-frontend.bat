@echo off
echo ========================================
echo   Fixing React Frontend Styles
echo ========================================
echo.

cd frontend-react

echo Checking if node_modules exists...
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
) else (
    echo Dependencies already installed.
)

echo.
echo Rebuilding CSS...
npm run build

echo.
echo Starting development server...
echo Frontend will be available at: http://localhost:3000
echo.

npm run dev
