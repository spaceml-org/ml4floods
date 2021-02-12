
# SPOTDOT Docker Container
# MCMC Runs

# NEED A DOCKER IMAGE
FROM nvidia/cuda:10.2-base-ubuntu18.04

# ENVIRONMENT NAME
ENV ML4FLOODS=0.1

# DEFINE PACKAGES
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    bash \
    git \
    gcc \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
COPY . /app
WORKDIR /app

# Add permissions
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user


ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh

# Install Conda Environment 
RUN conda env update -f environments/environment.yml -n base --prune

#  Check environments and versions
ENV PYTHONPATH="/app"
RUN sh -c 'conda --version'
RUN sh -c 'which python'

# RUN STUFF
RUN sh -c 'echo -e IMAGE COMPLETED - READY TO RUN'

# Keep Docker Image Open
CMD tail -f /dev/null