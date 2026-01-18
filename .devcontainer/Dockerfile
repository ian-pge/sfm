# ------------------------------------------------------------------------------
# Base image: Ubuntu 22.04
# ------------------------------------------------------------------------------
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# Install Basic System Dependencies (Curl needed for Pixi)
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
  curl \
  git \
  build-essential \
  ca-certificates \
  sudo \
  wget \
  && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Setup User
# ------------------------------------------------------------------------------
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=zed

RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
  useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
  echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
  chmod 0440 /etc/sudoers.d/${USER_NAME}
