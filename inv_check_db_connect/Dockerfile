FROM jupyter/base-notebook
WORKDIR /home/jovyan/workspace
COPY requirements.txt requirements.txt
RUN conda install --quiet --yes --file requirements.txt
