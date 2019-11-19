FROM jupyter/datascience-notebook
MAINTAINER Yusuke Takahashi

RUN pip install jupyterlab
RUN jupyter serverextension enable --py jupyterlab
RUN pip install tensorflow-gpu
RUN pip install tqdm

RUN jupyter labextension install @lckr/jupyterlab_variableinspector

RUN jupyter labextension install @jupyterlab/toc

RUN pip install tensorboard 
RUN pip install jupyter-tensorboard 
# RUN jupyter labextension install jupyterlab_tensorboard
# RUN jupyter serverextension enable --py jupyterlab_tensorboard

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"


ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
