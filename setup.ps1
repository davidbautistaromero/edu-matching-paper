# setup.ps1 — Entorno de desarrollo paper-AI
# Crea el .venv e instala todas las dependencias necesarias para correr el pipeline visual.
#
# Uso:
#   cd C:\paper-AI
#   .\setup.ps1
#
# Requisitos previos:
#   - Python 3.10+ instalado y en el PATH
#   - El checkpoint DeepLabV3+ en checkpoints\ (ver README para descarga)

Set-Location $PSScriptRoot

Write-Host "=== paper-AI setup ===" -ForegroundColor Cyan

# 1. Crear .venv si no existe
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] .venv ya existe, omitiendo creacion." -ForegroundColor Green
}

# 2. Actualizar pip
Write-Host "[2/4] Actualizando pip..." -ForegroundColor Yellow
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

# 3. Instalar torch con soporte GPU si hay CUDA, si no CPU
Write-Host "[3/4] Detectando GPU..." -ForegroundColor Yellow
$cudaVersion = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
if ($cudaVersion) {
    Write-Host "GPU detectada. Instalando torch con CUDA 12.4..." -ForegroundColor Green
    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
} else {
    Write-Host "Sin GPU. Instalando torch CPU..." -ForegroundColor Yellow
    .venv\Scripts\pip install torch torchvision --quiet
}

# Instalar resto de dependencias
Write-Host "Instalando resto de dependencias (requirements.txt)..." -ForegroundColor Yellow
.venv\Scripts\pip install -r requirements.txt --quiet

# Instalar CLIP (no está en PyPI, requiere instalación desde GitHub)
Write-Host "Instalando CLIP (OpenAI)..." -ForegroundColor Yellow
.venv\Scripts\pip install git+https://github.com/openai/CLIP.git --quiet

# 4. Verificar instalacion de torch
Write-Host "[4/4] Verificando torch..." -ForegroundColor Yellow
$torchOk = .venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| device:', 'cuda' if torch.cuda.is_available() else 'cpu')" 2>&1
Write-Host $torchOk -ForegroundColor Green

# 5. Verificar y descargar checkpoint si falta
Write-Host ""
Write-Host "--- Checkpoint ---" -ForegroundColor Cyan
$ckpt = "checkpoints\best_deeplabv3plus_resnet101_cityscapes_os16.pth"
if (Test-Path $ckpt) {
    $mb = [math]::Round((Get-Item $ckpt).Length / 1MB, 0)
    Write-Host "OK: $ckpt ($mb MB)" -ForegroundColor Green
} else {
    Write-Host "Checkpoint no encontrado. Descargando (~449 MB)..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path checkpoints | Out-Null
    .venv\Scripts\pip install gdown --quiet
    .venv\Scripts\python.exe -m gdown 1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb -O $ckpt
    if (Test-Path $ckpt) {
        $mb = [math]::Round((Get-Item $ckpt).Length / 1MB, 0)
        Write-Host "Descargado: $ckpt ($mb MB)" -ForegroundColor Green
    } else {
        Write-Host "ERROR: no se pudo descargar el checkpoint." -ForegroundColor Red
        Write-Host "Descargalo manualmente desde Google Drive:" -ForegroundColor Yellow
        Write-Host "  https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb"
        Write-Host "y guárdalo en: $ckpt"
    }
}

Write-Host ""
Write-Host "=== Setup completo ===" -ForegroundColor Cyan
Write-Host "Para correr el pipeline:"
Write-Host "  .venv\Scripts\python.exe scripts\02b_seg_cityscapes.py"
