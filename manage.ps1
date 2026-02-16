param(
    [string]$Command
)

$PythonStr = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonStr = "py"
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonStr = "python3"
}

if ($Command -eq "install") {
    & $PythonStr -m pip install -r requirements.txt
}
elseif ($Command -eq "sim") {
    & $PythonStr simulations/src/main.py
}
elseif ($Command -eq "paper") {
    Push-Location manuscript/tex
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
    if (-not (Test-Path ../build)) {
        New-Item -ItemType Directory -Path ../build | Out-Null
    }
    Copy-Item main.pdf ../build/main.pdf -Force
    Pop-Location
}
elseif ($Command -eq "clean") {
    if (Test-Path manuscript/build) {
        Remove-Item -Recurse -Force manuscript/build/* -ErrorAction SilentlyContinue
    }
    Remove-Item -Path "manuscript/tex/*.aux", "manuscript/tex/*.bbl", "manuscript/tex/*.blg", "manuscript/tex/*.log", "manuscript/tex/*.out", "manuscript/tex/*.synctex.gz" -ErrorAction SilentlyContinue
}
else {
    Write-Host "Usage: ./manage.ps1 [install|sim|paper|clean]"
}
