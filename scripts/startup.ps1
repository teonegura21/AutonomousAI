param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [bool]$OpenBrowser = $true,
  [switch]$SkipDocker,
  [switch]$SkipOllama,
  [switch]$SkipBackend,
  [switch]$SkipFrontend,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Section {
  param([string]$Title)
  Write-Host ""
  Write-Host ("=" * 72)
  Write-Host $Title
  Write-Host ("=" * 72)
}

function Command-Exists {
  param([string]$Name)
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-Url {
  param([string]$Url, [int]$TimeoutSec = 2)
  try {
    Invoke-RestMethod -Uri $Url -TimeoutSec $TimeoutSec -Method Get | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Wait-ForUrl {
  param([string]$Url, [int]$Retries = 8, [int]$DelaySec = 1)
  for ($i = 0; $i -lt $Retries; $i++) {
    if (Test-Url $Url) {
      return $true
    }
    Start-Sleep -Seconds $DelaySec
  }
  return $false
}

function Test-Port {
  param([int]$Port)
  try {
    return Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet
  } catch {
    return $false
  }
}

function Wait-ForPort {
  param([int]$Port, [int]$Retries = 10, [int]$DelaySec = 1)
  for ($i = 0; $i -lt $Retries; $i++) {
    if (Test-Port $Port) {
      return $true
    }
    Start-Sleep -Seconds $DelaySec
  }
  return $false
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

$python = $null
if (Test-Path (Join-Path $root "venv\Scripts\python.exe")) {
  $python = Join-Path $root "venv\Scripts\python.exe"
} elseif (Command-Exists "python") {
  $python = "python"
}

Write-Section "Environment"
Write-Host ("Workspace: " + $root)
Write-Host ("Python: " + ($python ?? "not found"))
Write-Host ("Docker: " + (Command-Exists "docker"))
Write-Host ("Node: " + (Command-Exists "node"))
Write-Host ("npm: " + (Command-Exists "npm"))

if (-not $SkipDocker) {
  Write-Section "Docker Stack"
  if (-not (Command-Exists "docker")) {
    Write-Warning "Docker not installed. Skipping docker checks."
    $SkipDocker = $true
  }
  if (-not $SkipDocker) {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
      Write-Warning "Docker is not running. Start Docker Desktop, then re-run."
      $SkipDocker = $true
    }
  }
  if (-not $SkipDocker) {
    $composeFile = Join-Path $root "docker\docker-compose.yml"
    if (Test-Path $composeFile) {
      if (Command-Exists "docker-compose") {
        docker-compose -f $composeFile up -d
      } else {
        docker compose -f $composeFile up -d
      }
      Write-Host "Docker services are up."
    } else {
      Write-Warning "docker-compose.yml not found. Skipping docker services."
    }
  }
}

if (-not $SkipOllama) {
  Write-Section "Ollama"
  $ollamaBase = "http://localhost:11434"
  $cfgPath = Join-Path $root "config\settings.yaml"
  if ($python -and (Test-Path $cfgPath)) {
    try {
      $ollamaBase = & $python -c "import yaml; p=r'config/settings.yaml'; d=yaml.safe_load(open(p, 'r', encoding='utf-8')) or {}; base=d.get('providers',{}).get('ollama',{}).get('api_base'); print(base or 'http://localhost:11434')"
    } catch {
      $ollamaBase = "http://localhost:11434"
    }
  }
  $ollamaTags = "$ollamaBase/api/tags"
  if (Test-Url $ollamaTags) {
    Write-Host ("Ollama OK at " + $ollamaBase)
  } else {
    Write-Warning ("Ollama not responding at " + $ollamaBase)
    if (Command-Exists "ollama") {
      Write-Host "Starting Ollama service..."
      Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
      if (Wait-ForUrl $ollamaTags) {
        Write-Host "Ollama started."
      } else {
        Write-Warning "Ollama still not responding."
      }
    } elseif (-not $SkipDocker -and (Command-Exists "docker")) {
      $containers = docker ps -a --format "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}"
      $ollamaContainers = @()
      foreach ($line in $containers) {
        $parts = $line -split "\|"
        if ($parts.Length -ge 4 -and ($parts[1] -match "ollama" -or $parts[2] -match "ollama")) {
          $ollamaContainers += [pscustomobject]@{
            Id = $parts[0]
            Name = $parts[1]
            Image = $parts[2]
            Status = $parts[3]
          }
        }
      }
      if ($ollamaContainers.Count -gt 0) {
        foreach ($c in $ollamaContainers) {
          if ($c.Status -notmatch "^Up") {
            Write-Host ("Starting Ollama container: " + $c.Name)
            docker start $c.Id | Out-Null
          }
        }
        if (Wait-ForUrl $ollamaTags) {
          Write-Host "Ollama started via Docker."
        } else {
          Write-Warning "Ollama container is running but API not responding."
        }
      } else {
        Write-Warning "No Ollama container found."
      }
    } else {
      Write-Warning "Ollama CLI not found. Install Ollama or run it in Docker."
    }
  }
}

if (-not $SkipBackend) {
  Write-Section "Backend"
  if (-not $python) {
    Write-Warning "Python not available. Skipping backend."
  } elseif (Test-Port $BackendPort) {
    Write-Warning ("Port " + $BackendPort + " already in use. Backend not started.")
  } else {
    $backendArgs = @(
      "-m", "uvicorn",
      "multi_agent_framework.ui.web.backend:app",
      "--host", "127.0.0.1",
      "--port", $BackendPort
    )
    Start-Process -FilePath $python -ArgumentList $backendArgs -WorkingDirectory $root
    if (Wait-ForUrl ("http://127.0.0.1:" + $BackendPort + "/api/health")) {
      Write-Host ("Backend running at http://127.0.0.1:" + $BackendPort)
    } else {
      Write-Warning "Backend did not respond yet."
    }
  }
}

if (-not $SkipFrontend) {
  Write-Section "Frontend"
  if (-not (Command-Exists "npm")) {
    Write-Warning "npm not available. Skipping frontend."
  } else {
    $frontendDir = Join-Path $root "multi_agent_framework\ui\web\frontend"
    if (-not $SkipInstall) {
      if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
        Write-Host "Installing frontend dependencies..."
        Push-Location $frontendDir
        npm install
        Pop-Location
      }
    }
    if (Test-Port $FrontendPort) {
      Write-Warning ("Port " + $FrontendPort + " already in use. Frontend not started.")
    } else {
      $env:VITE_API_BASE = "http://127.0.0.1:$BackendPort"
      Start-Process -FilePath "npm" -ArgumentList "run", "dev", "--", "--host", "127.0.0.1", "--port", $FrontendPort -WorkingDirectory $frontendDir
      Remove-Item Env:\VITE_API_BASE -ErrorAction SilentlyContinue
      Write-Host ("Frontend starting at http://127.0.0.1:" + $FrontendPort)
    }

    if ($OpenBrowser) {
      $frontendUrl = "http://127.0.0.1:$FrontendPort"
      if (Wait-ForPort $FrontendPort) {
        Start-Process $frontendUrl
      } else {
        Write-Warning "Frontend port did not respond; browser not opened."
      }
    }
  }
}

Write-Section "Done"
Write-Host "If something did not start, scroll up for warnings."
