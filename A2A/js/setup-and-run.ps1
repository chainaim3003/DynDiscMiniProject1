# Quick Setup and Run Script
# Save this as setup-and-run.ps1

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  A2A Agentic Negotiation System - Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location -Path "C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js"

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found!" -ForegroundColor Yellow
    Write-Host "Creating .env from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ Created .env file" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit .env and add your OpenAI API key!" -ForegroundColor Yellow
    Write-Host "   File location: C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js\.env" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to open .env file in notepad..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    notepad ".env"
    Write-Host ""
    Write-Host "After adding your API key, press any key to continue..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pnpm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error installing dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host ""

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Ready to start negotiation!" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting agents in separate windows..." -ForegroundColor Cyan
Write-Host ""

# Start Seller Agent
Write-Host "🏪 Starting Seller Agent (Port 8080)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js; pnpm run agents:seller"
Start-Sleep -Seconds 3

# Start Buyer Agent
Write-Host "🛒 Starting Buyer Agent (Port 9090)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js; pnpm run agents:buyer"
Start-Sleep -Seconds 3

# Start CLI
Write-Host "💬 Starting CLI..." -ForegroundColor Green
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\CHAINAIM3003\mcp-servers\algoTITANV6\Legent\A2A\js; pnpm run a2a:cli http://localhost:9090"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  All agents started!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "In the CLI window, type: start negotiation" -ForegroundColor Yellow
Write-Host ""
Write-Host "Watch the negotiation unfold across all windows!" -ForegroundColor Cyan
Write-Host ""
