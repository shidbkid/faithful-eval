$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Sid\faithful-eval"
$env:NLTK_DISABLE_IMPORT_SECURITY = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".\.venv\Scripts\python.exe"

Write-Host "=== 1/3 Full RAGTruth Summary with preds ===" -ForegroundColor Cyan
& $py run.py --dataset ragtruth --out results-ragtruth.json --save-preds
if ($LASTEXITCODE -ne 0) { throw "step 1 failed" }

Write-Host "=== 2/3 Judge scaling curve ===" -ForegroundColor Cyan
& $py run.py --dataset ragtruth --judge-model 0.5b,1.5b,3b,7b-4bit --out results-scale.json --save-preds
if ($LASTEXITCODE -ne 0) { throw "step 2 failed" }

Write-Host "=== 3/3 RAGTruth QA confirmation (n=400) ===" -ForegroundColor Cyan
& $py run.py --dataset ragtruth --task QA --limit 400 --only random,rouge-l,nli-deberta,llm-judge --out results-ragtruth-qa.json --save-preds
if ($LASTEXITCODE -ne 0) { throw "step 3 failed" }

Write-Host "=== analyze + plot ===" -ForegroundColor Cyan
& $py analyze.py --preds results-ragtruth.preds.json
& $py plot.py

Write-Host "ALL UPGRADES DONE" -ForegroundColor Green
