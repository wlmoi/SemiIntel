# GitHub Setup and Deployment Script for SEMIINTEL
# Run this after creating your GitHub repository

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   SEMIINTEL - GitHub Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitInstalled) {
    Write-Host "ERROR: Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    pause
    exit
}

Write-Host "Git is installed: " -NoNewline
git --version
Write-Host ""

# Check if this is already a git repository
if (Test-Path ".git") {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Configure Git User" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Get current git config
$currentName = git config user.name
$currentEmail = git config user.email

if ($currentName) {
    Write-Host "Current Git Name: $currentName" -ForegroundColor Green
} else {
    Write-Host "Git user name not configured" -ForegroundColor Yellow
    $name = Read-Host "Enter your Git username"
    git config user.name "$name"
    Write-Host "✓ Git username set to: $name" -ForegroundColor Green
}

if ($currentEmail) {
    Write-Host "Current Git Email: $currentEmail" -ForegroundColor Green
} else {
    Write-Host "Git email not configured" -ForegroundColor Yellow
    $email = Read-Host "Enter your Git email"
    git config user.email "$email"
    Write-Host "✓ Git email set to: $email" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Deployment Files Created" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ requirements.txt (updated with Streamlit)" -ForegroundColor Green
Write-Host "✓ .streamlit/config.toml (theme & server config)" -ForegroundColor Green
Write-Host "✓ packages.txt (system dependencies)" -ForegroundColor Green
Write-Host "✓ DEPLOYMENT.md (comprehensive guide)" -ForegroundColor Green
Write-Host "✓ LICENSE (MIT License)" -ForegroundColor Green
Write-Host "✓ README.md (updated with deployment info)" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor Yellow
Write-Host "   Go to: https://github.com/new" -ForegroundColor White
Write-Host "   Repository name: SemiIntel" -ForegroundColor White
Write-Host "   Make it Public (required for free Streamlit Cloud)" -ForegroundColor White
Write-Host ""

Write-Host "2. Copy your repository URL from GitHub" -ForegroundColor Yellow
Write-Host "   Example: https://github.com/YOUR_USERNAME/SemiIntel.git" -ForegroundColor White
Write-Host ""

$continue = Read-Host "Have you created the GitHub repository? (y/n)"

if ($continue -eq "y" -or $continue -eq "Y") {
    Write-Host ""
    $repoUrl = Read-Host "Paste your GitHub repository URL"
    
    # Check if remote already exists
    $remoteExists = git remote get-url origin 2>$null
    
    if ($remoteExists) {
        Write-Host "Remote 'origin' already exists. Updating..." -ForegroundColor Yellow
        git remote set-url origin $repoUrl
    } else {
        git remote add origin $repoUrl
    }
    
    Write-Host "✓ Remote repository configured" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Adding files to git..." -ForegroundColor Yellow
    git add .
    Write-Host "✓ Files staged" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Creating commit..." -ForegroundColor Yellow
    git commit -m "Initial commit: SEMIINTEL web application with deployment config"
    Write-Host "✓ Commit created" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    Write-Host "(You may need to authenticate with GitHub)" -ForegroundColor Cyan
    
    try {
        git push -u origin main 2>&1
        if ($LASTEXITCODE -ne 0) {
            # Try master branch if main fails
            Write-Host "Trying 'master' branch..." -ForegroundColor Yellow
            git branch -M main
            git push -u origin main
        }
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "   SUCCESS! Code pushed to GitHub" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
    } catch {
        Write-Host ""
        Write-Host "Push failed. You may need to authenticate." -ForegroundColor Red
        Write-Host "Try running manually: git push -u origin main" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "   Deploy to Streamlit Cloud" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "3. Go to: https://share.streamlit.io" -ForegroundColor Yellow
    Write-Host "4. Sign in with your GitHub account" -ForegroundColor Yellow
    Write-Host "5. Click 'New app'" -ForegroundColor Yellow
    Write-Host "6. Select your repository: YOUR_USERNAME/SemiIntel" -ForegroundColor Yellow
    Write-Host "7. Set Main file: app.py" -ForegroundColor Yellow
    Write-Host "8. Click 'Deploy!'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Your app will be live in 2-5 minutes!" -ForegroundColor Green
    Write-Host ""
    Write-Host "For detailed instructions, see: DEPLOYMENT.md" -ForegroundColor Cyan
    
} else {
    Write-Host ""
    Write-Host "No problem! Here's what to do:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Go to https://github.com/new" -ForegroundColor White
    Write-Host "2. Create repository named 'SemiIntel' (Public)" -ForegroundColor White
    Write-Host "3. Run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run these commands manually:" -ForegroundColor Cyan
    Write-Host "  git add ." -ForegroundColor White
    Write-Host "  git commit -m 'Initial commit'" -ForegroundColor White
    Write-Host "  git remote add origin YOUR_REPO_URL" -ForegroundColor White
    Write-Host "  git push -u origin main" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "For complete deployment guide, see:" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT.md" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

pause
