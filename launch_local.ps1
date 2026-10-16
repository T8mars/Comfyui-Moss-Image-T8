param(
    [Parameter(Mandatory=$true)][string]$ComfyRoot,
    [string]$Python = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    [int]$Port = 8279
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$comfyMain = Join-Path $ComfyRoot 'main.py'
if (-not (Test-Path -LiteralPath $comfyMain)) { throw "ComfyUI main.py missing: $comfyMain" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python missing: $Python. Pass -Python with the prepared environment." }
$localBase = Join-Path $projectRoot '.comfy'
$modelsPath = Join-Path $projectRoot 'models'
$customPath = Join-Path $localBase 'custom_nodes'
New-Item -ItemType Directory -Path $customPath,$modelsPath -Force | Out-Null
$pluginPath = Join-Path $customPath 'ComfyUI-MOSS-Image'
if (-not (Test-Path -LiteralPath $pluginPath)) {
    New-Item -ItemType Junction -Path $pluginPath -Target $projectRoot | Out-Null
}
else {
    $pluginItem = Get-Item -LiteralPath $pluginPath -Force
    $targetPaths = @($pluginItem.Target)
    if (-not ($pluginItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        $targetPaths.Count -ne 1 -or
        [IO.Path]::GetFullPath($targetPaths[0]).TrimEnd('\') -ne $projectRoot.TrimEnd('\')) {
        throw "Custom-node path points to another checkout: $pluginPath. Inspect it and remove the stale link before retrying."
    }
}
$comfyArgs = @('-s','-u',$comfyMain,'--base-directory',$localBase,'--models-directory',$modelsPath,
    '--port',"$Port",'--listen','127.0.0.1','--disable-auto-launch','--offline',
    '--use-pytorch-cross-attention','--disable-dynamic-vram','--cache-classic','--database-url','sqlite:///:memory:')
& $Python @comfyArgs
exit $LASTEXITCODE
