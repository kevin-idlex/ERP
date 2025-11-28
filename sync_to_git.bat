@echo off
REM ============================================================================
REM IdleX ERP - Sync to Git Repository
REM ============================================================================
REM This script copies all ERP files from Dropbox to your Git repository
REM and opens VS Code for you to review and commit changes.
REM ============================================================================

setlocal enabledelayedexpansion

REM --- CONFIGURATION ---
set "DROPBOX_DIR=D:\Dropbox\IdleX\IdleX_ERP"
set "GIT_REPO=D:\Projects\IdleX_ERP"
REM Change GIT_REPO to wherever your Git repository is located

REM --- Colors (Windows 10+) ---
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "RESET=[0m"

echo.
echo %GREEN%============================================================================%RESET%
echo %GREEN%  IdleX ERP - Sync to Git Repository%RESET%
echo %GREEN%============================================================================%RESET%
echo.

REM --- Check source directory ---
if not exist "%DROPBOX_DIR%" (
    echo %RED%ERROR: Dropbox directory not found: %DROPBOX_DIR%%RESET%
    echo Please update DROPBOX_DIR in this script.
    pause
    exit /b 1
)

REM --- Create Git repo directory if needed ---
if not exist "%GIT_REPO%" (
    echo %YELLOW%Creating Git repository directory: %GIT_REPO%%RESET%
    mkdir "%GIT_REPO%"
    
    echo %YELLOW%Initializing Git repository...%RESET%
    cd /d "%GIT_REPO%"
    git init
    
    REM Create .gitignore
    echo # Python > .gitignore
    echo __pycache__/ >> .gitignore
    echo *.py[cod] >> .gitignore
    echo *.so >> .gitignore
    echo .Python >> .gitignore
    echo venv/ >> .gitignore
    echo .env >> .gitignore
    echo. >> .gitignore
    echo # Database >> .gitignore
    echo *.db >> .gitignore
    echo *.sqlite >> .gitignore
    echo. >> .gitignore
    echo # IDE >> .gitignore
    echo .idea/ >> .gitignore
    echo .vscode/ >> .gitignore
    echo *.swp >> .gitignore
    echo. >> .gitignore
    echo # OS >> .gitignore
    echo .DS_Store >> .gitignore
    echo Thumbs.db >> .gitignore
    
    echo %GREEN%Created .gitignore%RESET%
)

REM --- List files to sync ---
echo %YELLOW%Files in Dropbox:%RESET%
echo.
dir /b "%DROPBOX_DIR%\*.py" 2>nul
dir /b "%DROPBOX_DIR%\*.txt" 2>nul
dir /b "%DROPBOX_DIR%\*.md" 2>nul
dir /b "%DROPBOX_DIR%\*.png" 2>nul
dir /b "%DROPBOX_DIR%\Dockerfile" 2>nul
dir /b "%DROPBOX_DIR%\requirements.txt" 2>nul
echo.

REM --- Copy Python files ---
echo %YELLOW%Copying Python files...%RESET%
copy /y "%DROPBOX_DIR%\dashboard.py" "%GIT_REPO%\" 2>nul && echo   dashboard.py
copy /y "%DROPBOX_DIR%\seed_db.py" "%GIT_REPO%\" 2>nul && echo   seed_db.py

REM --- Copy supporting files ---
echo %YELLOW%Copying supporting files...%RESET%
copy /y "%DROPBOX_DIR%\requirements.txt" "%GIT_REPO%\" 2>nul && echo   requirements.txt
copy /y "%DROPBOX_DIR%\Dockerfile" "%GIT_REPO%\" 2>nul && echo   Dockerfile
copy /y "%DROPBOX_DIR%\README.md" "%GIT_REPO%\" 2>nul && echo   README.md

REM --- Copy logo files ---
echo %YELLOW%Copying logo files...%RESET%
copy /y "%DROPBOX_DIR%\logo_white.png" "%GIT_REPO%\" 2>nul && echo   logo_white.png
copy /y "%DROPBOX_DIR%\logo_blue.png" "%GIT_REPO%\" 2>nul && echo   logo_blue.png
copy /y "%DROPBOX_DIR%\icon_white.png" "%GIT_REPO%\" 2>nul && echo   icon_white.png

REM --- Copy tests folder ---
if exist "%DROPBOX_DIR%\tests" (
    echo %YELLOW%Copying tests folder...%RESET%
    if not exist "%GIT_REPO%\tests" mkdir "%GIT_REPO%\tests"
    xcopy /y /q "%DROPBOX_DIR%\tests\*.py" "%GIT_REPO%\tests\" 2>nul
    echo   tests\*.py
)

echo.
echo %GREEN%============================================================================%RESET%
echo %GREEN%  Sync Complete!%RESET%
echo %GREEN%============================================================================%RESET%
echo.

REM --- Show Git status ---
cd /d "%GIT_REPO%"
echo %YELLOW%Git Status:%RESET%
echo.
git status --short
echo.

REM --- Open VS Code ---
echo %YELLOW%Opening VS Code...%RESET%
code "%GIT_REPO%"

echo.
echo %GREEN%Next steps:%RESET%
echo   1. Review changes in VS Code
echo   2. Open terminal in VS Code (Ctrl+`)
echo   3. Stage changes:    git add .
echo   4. Commit:           git commit -m "Your message"
echo   5. Push:             git push origin main
echo.

pause
