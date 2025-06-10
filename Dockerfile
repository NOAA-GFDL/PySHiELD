FROM python:3.11

RUN apt-get update &&\
    apt install -y --no-install-recommends \
    software-properties-common

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends\
    g++ \
    gcc \
    gfortran \
    git \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libopenmpi3 \
    libopenmpi-dev \
    libhdf5-serial-dev \
    libffi-dev \
    libssl-dev \
    netcdf-bin \
    libnetcdf-dev

RUN wget -O - https://www.openssl.org/source/openssl-1.1.1u.tar.gz | tar zxf - && \
    cd openssl-1.1.1u && \
    ./config --prefix=/usr/local

# RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -O ~/miniforge.sh && \
#     mkdir -p /root/.conda && \
#     bash miniconda.sh -b -p /root/miniconda3 && \
#     rm -f miniconda.sh

COPY --from=continuumio/miniconda3:4.12.0 /opt/conda /opt/conda

ENV PATH=/opt/conda/bin:$PATH

# Use conda to unstall RTE-RRTMGP
RUN set -ex && \
    conda config --set always_yes yes --set changeps1 no && \
    conda config --set ssl_verify false && \
    conda config --env --set subdir linux-64 && \
    conda info -a && \
    conda config --append channels conda-forge && \
    conda install --quiet --freeze-installed -c main conda-pack && \
    conda install -c conda-forge ninja && \
    conda install -vv -c conda-forge rte_rrtmgp

# Install pyrte_rrtmgp via conda
# RUN conda install -vv -c conda-forge pyrte_rrtmgp
# RUN conda install conda-forgwe::pyrte_rrtmgp

RUN python3 -m pip install --upgrade setuptools pip wheel

# Check python & pip
RUN python --version
RUN which python
RUN pip --version
RUN which pip

COPY ./ /pySHiELD/

# Install pySHiELD and the full dependencies
RUN pip install -e pySHiELD[develop]

RUN git clone -b versions https://github.com/oelbert/pyRTE-RRTMGP.git && \
    cd pyRTE-RRTMGP && \ 
    pip install -e .

RUN pip install \
    matplotlib \
    cython \
    cartopy \
    ipyparallel \
    jupyter \
    jupyterlab \
    shapely \
    jupyterlab_code_formatter \
    mpi4py \
    pytest \
    pytest-subtests \
    pytest-regressions \
    pytest-profiling \
    pytest-cov

# # set up for fv3viz
RUN cd /
RUN git clone --recursive https://github.com/ai2cm/fv3net.git
RUN cd fv3net && git checkout 1d168ef
RUN pip install fv3net/external/vcm
ENV PYTHONPATH=/fv3net/external/fv3viz

ENV CFLAGS="-I/usr/include -DACCEPT_USE_OF_DEPRECATED_PROJ_API_H=1"

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
