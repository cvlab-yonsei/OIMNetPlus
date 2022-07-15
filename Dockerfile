FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0

RUN pip install jupyter jupyterlab numpy scipy ipython pandas easydict torchsummary tensorboard seaborn tabulate yacs
RUN pip install tensorboardX --use-feature=2020-resolver
RUN pip install torchvision==0.8.2
RUN pip install opencv-python ipykernel
RUN pip install sklearn

CMD ["/bin/bash"]