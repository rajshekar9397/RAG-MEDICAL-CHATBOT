FROM --platform=linux/amd64 jenkins/jenkins:lts

USER root

# Install docker CLI only (not full engine)
RUN apt-get update && \
    apt-get install -y docker.io && \
    apt-get clean

RUN usermod -aG docker jenkins

USER jenkins
