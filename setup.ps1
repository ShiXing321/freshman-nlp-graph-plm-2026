param([string]$EnvironmentPath = '.venv-reproduce')
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath $EnvironmentPath) {
    throw "目录已存在：$EnvironmentPath。请换一个新环境名称。"
}
py -3.11 -m venv $EnvironmentPath
if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败' }
$taskPython = Join-Path $EnvironmentPath 'Scripts\python.exe'
& $taskPython -m pip install --upgrade pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple
if ($LASTEXITCODE -ne 0) { throw '升级pip失败' }
$taskCudaWheel = Join-Path $PSScriptRoot '.cache\wheels\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl'
if (Test-Path -LiteralPath $taskCudaWheel) {
    & $taskPython -m pip install $taskCudaWheel --index-url https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    & $taskPython -m pip install 'https://s3.jcloud.sjtu.edu.cn/899a892efef34b1b944a19981040f55b-oss01/pytorch-wheels/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl' --index-url https://pypi.tuna.tsinghua.edu.cn/simple
}
if ($LASTEXITCODE -ne 0) { throw '安装CUDA PyTorch失败' }
& $taskPython -m pip install -r requirements-lock.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
if ($LASTEXITCODE -ne 0) { throw '安装依赖失败，请检查网络与错误输出' }
& $taskPython -c 'import torch, torch_geometric, transformers; print(torch.__version__); print("CUDA可用：", torch.cuda.is_available())'
Write-Output "复现环境已建立。请把报告命令中的 .venv 换为 $EnvironmentPath。"
