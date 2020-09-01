FROM ubuntu:18.04

# Python 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        git \
        curl \
        wget \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        unzip

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN git clone https://github.com/DolotovEvgeniy/cat-vs-dog-tensorflow.git
WORKDIR "/cat-vs-dog-tensorflow"
RUN pip install -r requirements.txt

RUN chmod +x prepare_dataset.sh && \
    ./prepare_dataset.sh

RUN mkdir checkpoints
