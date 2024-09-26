FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
      curl \
      git \
      wget \
      bash \
      vim \
      pip \
      pkg-config \
      default-jdk \
      libgl1-mesa-glx \
      libegl1-mesa \
      libxrandr2 \
      libxrandr2 \
      libxss1 \
      libxcursor1 \
      libxcomposite1 \
      libasound2 \
      libxi6 \
      libxtst6 \
      libmysqlclient-dev \
      ;

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3.10 /usr/bin/python; fi && \
    if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.1 \
    torchaudio==2.0.1 \
    huggingface-hub==0.20.2 \
    matplotlib==3.7.0 \
    psutil==5.9.4 \
    pyyaml==6.0 \
    regex==2022.10.31 \
    tokenizers==0.15.0 \
    tqdm==4.64.1 \
    timm==0.6.13 \
    webdataset==0.2.48 \
    opencv-python==4.7.0.72 \
    decord==0.6.0 \
    peft==0.2.0 \
    sentence-transformers \
    gradio==3.47.1 \
    accelerate==0.26.1 \
    bitsandbytes==0.37.0 \
    scikit-image \
    visual-genome \
    pandas \
    shortuuid \
    chardet \
    openai \
    supervision \
    addict \
    yapf \
    pycocotools \
    pycocoevalcap \
    icecream \
    nltk \
    diffusers \
    compel \
    hpsv2==1.2.0 \
    prettytable \
    omegaconf \
    iopath \
    pattern


WORKDIR /root/share
RUN python -m nltk.downloader punkt_tab -d /root/nltk_data
