param(
    [ValidateSet("lexical", "dense", "hybrid", "hybrid_rerank")]
    [string]$Strategy = "lexical"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

function Invoke-Demo([string]$Token, [string]$Question, [string]$Department = "") {
    $arguments = @("-m", "acmeflow_rag.demo", "--strategy", $Strategy, "--token", $Token)
    if ($Department) { $arguments += @("--department", $Department) }
    $arguments += @("--question", $Question)
    & $pythonCommand @arguments
    if ($LASTEXITCODE -ne 0) { throw "Demo command failed with exit code $LASTEXITCODE" }
}

Write-Host "=== Grounded answer ==="
Invoke-Demo "dev-support-token" "Does cancellation automatically refund the last charge, and what must Support verify?"

Write-Host "`n=== Authorization denial ==="
Invoke-Demo "dev-employee-token" "What approvals are required for production access?" "Security"

Write-Host "`n=== Unanswerable-query behavior ==="
Invoke-Demo "dev-employee-token" "What is AcmeFlow's daily meal reimbursement cap?"
