param(
    [string]$VenvPath = ".\.venv310",
    [string]$ModelPath = ".\src\model.h5",
    [int]$CameraIndex = 0
)

# Activate venv
$activate = Join-Path $VenvPath 'Scripts\Activate.ps1'
if (-not (Test-Path $activate)) {
    Write-Error "Virtual environment activation script not found at $activate. Create the venv first with Python 3.10."
    exit 1
}

Write-Host "Activating venv: $VenvPath"
& $activate

# Check model
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found: $ModelPath`nPlace your model.h5 in the src folder or pass -ModelPath <path>."
    exit 1
}

Write-Host "Starting display mode (model=$ModelPath, camera index=$CameraIndex). Press 'q', 'Q', ESC or Ctrl+C to stop."
# Run the script
python .\src\emotions.py --mode display
