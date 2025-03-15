# base image
FROM ubuntu:22.04

# environment variables
ENV PATH /home/vscode/.local/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV DEBCONF_NONINTERACTIVE_SEEN=true

# install packages
RUN apt update \
    && apt -y upgrade \
    && apt -y install --no-install-recommends build-essential \
    && apt -y install --no-install-recommends manpages-dev \
    && apt -y install --no-install-recommends software-properties-common \
    && apt -y install --no-install-recommends gcc-12 \
    && apt -y install --no-install-recommends g++-12 \
    && apt -y install --no-install-recommends python3.10 \
    && apt -y install --no-install-recommends python3-pip \
    && apt -y install --no-install-recommends gdb \
    && apt -y install --no-install-recommends libboost-all-dev \
    && apt -y install --no-install-recommends git \
    && apt -y install --no-install-recommends parallel \
    && apt -y install --no-install-recommends zsh \
    && apt autoremove -y \
    && apt clean -y

# zsh settings
RUN chsh -s /bin/zsh
RUN echo 'alias ll="ls -la"' >> /etc/zsh/zshrc \
    && echo 'alias gcc="gcc-12"' >> /etc/zsh/zshrc \
    && echo 'alias g++="g++-12"' >> /etc/zsh/zshrc \
    && echo 'PROMPT="%U%F{45}%~%f%u > "' >> /etc/zsh/zshrc \
    && echo 'bindkey "^[[1;5C" forward-word' >> /etc/zsh/zshrc \
    && echo 'bindkey "^[[1;5D" backward-word' >> /etc/zsh/zshrc \
    && echo 'bindkey "^[[3;5~" kill-word' >> /etc/zsh/zshrc \
    && echo 'bindkey "^H" backward-kill-word' >> /etc/zsh/zshrc

# install python packages
RUN pip install --upgrade pip
RUN pip install \
    numpy==1.24.1 \
    scipy==1.10.1 \
    networkx==3.0 \
    sympy==1.11.1 \
    sortedcontainers==2.4.0 \
    more-itertools==9.0.0 \
    shapely==2.0.0 \
    bitarray==2.6.2 \
    PuLP==2.7.0 \
    mpmath==1.2.1 \
    pandas==1.5.2 \
    z3-solver==4.12.1.0 \
    scikit-learn==1.3.0 \
    cppyy==2.4.1 \
    ruff \
    isort \
    black \
    tqdm
RUN pip install git+https://github.com/not522/ac-library-python
