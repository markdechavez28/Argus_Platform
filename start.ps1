#!/usr/bin/env pwsh
# Start Argus Platform with Docker Compose

Write-Host "Starting Argus Platform..." -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
$dockerRunning = docker info 2>$null
if (-not $dockerRunning) {
    Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Build and start services
Write-Host "Building and starting services..." -ForegroundColor Yellow
docker-compose up -d --build

# Wait for services to be ready
Write-Host ""
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host ""
Write-Host "Service Status:" -ForegroundColor Green
docker-compose ps

Write-Host ""
Write-Host "Argus Platform is running!" -ForegroundColor Green
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "   Argus Agent (Streamlit):  http://localhost:8501" -ForegroundColor White
Write-Host "   Prometheus Metrics:       http://localhost:8000/metrics" -ForegroundColor White
Write-Host "   Prometheus Dashboard:     http://localhost:9090" -ForegroundColor White
Write-Host "   Grafana Dashboard:        http://localhost:3000" -ForegroundColor White
Write-Host "   (Grafana default login: admin/admin)" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop: docker-compose down" -ForegroundColor Yellow
