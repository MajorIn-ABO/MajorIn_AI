# 가상 환경 설정 
$envList = conda env list
if ($envList -match "myenv") {
    Write-Output "myenv environment already exists. Activating..."
    conda activate myenv
} else {
    Write-Output "Creating myenv environment..."
    conda create -n myenv python=3.8 -y
    Write-Output "Activating myenv environment..."
    conda activate myenv
}

# conda 명령어 실행
conda install paddlepaddle==2.6.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ # cpu 버전
pip install "paddleocr>=2.0.1"
conda install fastapi -y
conda install "uvicorn[standard]" -y

