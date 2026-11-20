param(
    [string]$InstallRoot = 'J:\spaceteamPro\Main',
    [string]$Scene = 'Local/Repos/STU4-Sim/Sims/FINAL_SUBMISSION.simconf'
)

$ErrorActionPreference = 'Stop'
$client = Join-Path $InstallRoot 'SpaceCRAFT\SpaceCRAFT.exe'
$stp = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro'
$runtime = Join-Path $stp 'Versions\v0.56.0'
foreach ($required in @(
    $client
    (Join-Path $InstallRoot 'ReleaseVersion_0.56.0_Win64_User.txt')
    (Join-Path $runtime 'py\Scripts\python.exe')
    (Join-Path $stp 'Assets\Repos\STU4-Solution\Controller.py')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required Space Teams PRO file missing: $required"
    }
}
if ($Scene.StartsWith('Local/')) {
    $sceneFile = Join-Path (Join-Path $stp 'Assets') $Scene.Substring(6)
    if (-not (Test-Path -LiteralPath $sceneFile -PathType Leaf)) {
        throw "Simulation scene missing: $sceneFile"
    }
}

# Python 3.12 can fail to write bytecode through the cross-drive Assets junction.
$previousCache = $env:PYTHONPYCACHEPREFIX
try {
    $env:PYTHONPYCACHEPREFIX = Join-Path $runtime 'BatashaPyCache'
    Start-Process -FilePath $client -WorkingDirectory (Split-Path -Parent $client) -ArgumentList "--sim=`"$Scene`""
}
finally {
    $env:PYTHONPYCACHEPREFIX = $previousCache
}
