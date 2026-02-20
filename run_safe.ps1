# Wrapper to run Python scripts with limited threading to prevent system crashes
# Usage: ./run_safe.ps1 path/to/script.py [args]

$safeThreads = if ($env:SAFE_THREADS) { $env:SAFE_THREADS } else { "2" }

$env:OMP_NUM_THREADS = $safeThreads
$env:MKL_NUM_THREADS = $safeThreads
$env:OPENBLAS_NUM_THREADS = $safeThreads
$env:VECLIB_MAXIMUM_THREADS = $safeThreads
$env:NUMEXPR_NUM_THREADS = $safeThreads

Write-Host "Running in SAFE MODE (Limited to $safeThreads Threads)..." -ForegroundColor Yellow

# Use 'py' launcher if available, else 'python'
if (Get-Command py -ErrorAction SilentlyContinue) {
    py $args
}
else {
    python $args
}
