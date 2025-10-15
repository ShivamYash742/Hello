@echo off
echo Starting ISRO Thermal SR Application...

echo.
echo Starting Python API Server...
start "Python API" cmd /k "cd /d e:\Hello\thermal_sr && python api_server.py"

echo.
echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting Next.js Application...
start "Next.js App" cmd /k "cd /d e:\Hello && npm run dev"

echo.
echo Both servers are starting...
echo Python API: http://localhost:8000
echo Next.js App: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul