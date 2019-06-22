FROM python:3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m ipykernel.kernelspec
RUN python -m nltk.downloader stopwords

COPY guillotina_processing/* ./

RUN python setup.py develop

RUN mkdir /app
RUN mkdir /root/.jupyter

COPY jupyter_notebook_config.py /root/.jupyter/

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

CMD [ "jupyter", "notebook", "--notebook-dir=/app", "--ip=0.0.0.0", "--allow-root" ]
