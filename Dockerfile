FROM python:3.6.8-stretch

ENV LANG=C.UTF-8

WORKDIR /root
COPY . /root/

RUN pip3 install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

RUN apt-get update && apt-get install -y wget

RUN cd /root && mkdir tf_vae
RUN wget -O tf_vae/vae-300000.index 'https://docs.google.com/uc?export=download&id=1ulHdDxebH46m_0ZoLa2Wsz_6vStYqJQm'
RUN wget -O tf_vae/vae-300000.meta 'https://docs.google.com/uc?export=download&id=1nHN_i7Ro9g0lP4y_YQCvIWrOVX1I3CJa'
RUN wget -O tf_vae/vae-300000.data-00000-of-00001 'https://docs.google.com/uc?export=download&id=18rAJcUJwFJOAcjzsabtqK12udsHMZkVk'
RUN wget -O tf_vae/checkpoint 'https://docs.google.com/uc?export=download&id=18U4qMNBdyvEk-Y-Mr3MNPEHSHxhcO9hn'

RUN mkdir tf_gan3
RUN wget -O tf_gan3/gan-571445.meta 'https://docs.google.com/uc?export=download&id=15kEG1Tiu2FUg5SILVt_9yOsSd3QHwVGA'
RUN wget -O tf_gan3/gan-571445.index 'https://docs.google.com/uc?export=download&id=11uyFbQsRZoWa9Yq52AFXDXPjPQoGF_ER'
RUN wget -O tf_gan3/gan-571445.data-00000-of-00001 'https://docs.google.com/uc?export=download&id=11cbvz-CH3KvfZEwNQ2OUujfbf6AKNoQa'
RUN wget -O tf_gan3/checkpoint 'https://docs.google.com/uc?export=download&id=1A539u51t0L31Ab1M2uPUV2SsCFsNDQRo'


EXPOSE 8000

#ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/bin/bash"]

# docker build -t gswyhq/neural-painters -f Dockerfile .
