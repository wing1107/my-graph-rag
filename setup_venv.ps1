# 在项目根目录创建虚拟环境并安装依赖（PowerShell）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
# If pypi.org times out, use mirror:
# .\.venv\Scripts\pip.exe install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
.\.venv\Scripts\pip.exe install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "pip install failed."
    Write-Host "If you see 'No matching distribution' after repeated timeouts, the package index was unreachable (not a bad gradio version)."
    Write-Host "Retry on a stable network, use VPN/proxy, or run pip with another mirror (see comment above)."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "完成。激活虚拟环境后运行: python run_gradio.py"
Write-Host "  .\.venv\Scripts\Activate.ps1"
