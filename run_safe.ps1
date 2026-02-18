# Wrapper to run Python scripts with limited threading to prevent system crashes
# Usage: ./run_safe.ps1 path/to/script.py [args]

$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
$env:OPENBLAS_NUM_THREADS = "6"
$env:VECLIB_MAXIMUM_THREADS = "6"
$env:NUMEXPR_NUM_THREADS = "6"

Write-Host "Running in SAFE MODE (Limited to 6 Threads)..." -ForegroundColor Yellow

# Use 'py' launcher if available, else 'python'
if (Get-Command py -ErrorAction SilentlyContinue) {
    py $args
}
else {
    python $args
}
