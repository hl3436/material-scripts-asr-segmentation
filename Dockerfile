#FROM continuumio/miniconda3:latest
#RUN pip install torch torchvision tokenizers pytorch-lightning torchtext

FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

RUN pip install pytorch-lightning==0.9.0 sacremoses torchtext==0.8.0 tokenizers

ADD src /src
ADD scripts /scripts
ADD models /models 

VOLUME /input /output

#CMD echo "SUCCESS"
ENTRYPOINT ["bash","/scripts/segment.sh"]
