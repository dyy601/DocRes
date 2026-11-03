$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& 'D:\anaconda\envs\DocRes\python.exe' "$PSScriptRoot\train.py" --data_root "$PSScriptRoot\data\train" --batch_size 1 --im_size 256 --experiment_name docres_local @args
exit $LASTEXITCODE
