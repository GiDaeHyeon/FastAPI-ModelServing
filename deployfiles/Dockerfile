FROM nvidia/cuda:11.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV CKPT_DIR some/directory

RUN apt-get update
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip

COPY ./requirements.txt /usr/src/app/requirements.txt

WORKDIR /usr/src/app

RUN pip3 install -U pip
RUN pip3 install -U wheel
RUN pip3 install pip install -r requirements.txt
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /usr/src/app/fastapi_test