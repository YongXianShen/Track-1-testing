$ErrorActionPreference = "Stop"
$Image = "track1-local-v161"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputDir = Join-Path $Here "input"
$OutputDir = Join-Path $Here "output"
New-Item -ItemType Directory -Force $InputDir, $OutputDir | Out-Null
@'
[
  {"task_id":"p1","prompt":"What is the capital of Australia, and what body of water is it near?"},
  {"task_id":"p2","prompt":"A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?"}
]
'@ | Set-Content (Join-Path $InputDir "tasks.json") -Encoding UTF8

docker build --platform linux/amd64 -t $Image $Here
docker run --rm --platform linux/amd64 --memory 4g --cpus 2 `
  -v "${InputDir}:/input:ro" -v "${OutputDir}:/output" $Image
Get-Content (Join-Path $OutputDir "results.json")
