$ErrorActionPreference = "Stop"

$miktexBin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $miktexBin) {
    $env:Path = "$miktexBin;$env:Path"
}

Write-Host "Python:"
uv run python --version
Write-Host "`nCore packages:"
uv run python -c "import numpy, scipy, sklearn, pandas; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('scikit-learn', sklearn.__version__); print('pandas', pandas.__version__)"
Write-Host "`nTeX:"
xelatex --version | Select-Object -First 1
bibtex --version | Select-Object -First 1
Write-Host "`nTests:"
uv run pytest -q

