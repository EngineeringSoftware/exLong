# docker build -t etestgen .
# docker run -it etestgen /bin/bash

# Pull base image
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install sofware properties common
RUN apt-get update && \
    apt-get install -y software-properties-common
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && \
    apt-get -qq -y install apt-utils curl wget unzip zip gcc mono-mcs sudo less git build-essential pkg-config libicu-dev

# Add new user
RUN useradd -ms /bin/bash -c "Etest User" etest && echo "etest:etest" | chpasswd && adduser etest sudo
USER etest
WORKDIR /home/etest/

# Download and install SDKMAN
RUN curl -s "https://get.sdkman.io" | bash
# Add SDKMAN to PATH
ENV PATH="$HOME/.local/bin:$HOME/.sdkman/bin:${PATH}"
# Install JDK 8 and Maven
RUN bash -c "source $HOME/.sdkman/bin/sdkman-init.sh && sdk install java 8.0.302-open && sdk install maven 3.8.3"

# Set up working environment
# RUN bash -c "git clone https://github.com/EngineeringSoftware/etestgen.git etestgen"
ENV HOME=/home/etest
ENV USER=etest
COPY --chown=etest:etest . $HOME/etestgen-internal


# Install python etestgen
RUN cd $HOME/etestgen-internal \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && source $HOME/.local/bin/env \
    && uv venv \
    && uv sync

ENTRYPOINT /bin/bash
