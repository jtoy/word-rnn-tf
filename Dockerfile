#FROM b.gcr.io/tensorflow/tensorflow
FROM somatic/tensorflow-cpu
#RUN apt-get install -y 
#RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  
#RUN apt-get -y update
#RUN apt-get -y upgrade #TODO need to set the nvidia driver version manually so we dont get mismatch issues anymore
#mount the software, run initialize instructions; then remove from mount

