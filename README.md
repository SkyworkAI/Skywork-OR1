<div align="center">

# Skywork-o1

</div>


## Getting Started 🎯
### Installation

#### docker environment


```bash
docker pull whatcanyousee/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te2.0-megatron0.11.0-v0.0.6

# Launch the desired Docker image:
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v <image:tag>

# Inside the container, install Skywork-o1
git clone https://github.com/SkyworkAI/Skywork-o1.git && cd Skywork-o1 && pip3 install -e .
```

#### conda environment
```bash
# Installing Python 3.10 Environment.
conda create -n verl python==3.10
conda activate verl

# Installing RLLM dependencies.
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
git clone https://github.com/SkyworkAI/Skywork-o1.git
cd Skywork-o1
pip3 install -e .
```