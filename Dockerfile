# Use the official Ubuntu 20.04 image as the base image
FROM ubuntu:20.04

RUN apt-get update \
 && apt-get install -y sudo

RUN adduser --disabled-password --gecos '' alchemist
RUN adduser alchemist sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER alchemist

# this is where I was running into problems with the other approaches
RUN sudo apt-get update 