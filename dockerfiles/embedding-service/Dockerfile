from tensorflow/tensorflow

RUN echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial main restricted" > /etc/apt/sources.list \
  && echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial-updates main restricted" >> /etc/apt/sources.list \
  && echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial universe" >> /etc/apt/sources.list \
  && echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial-updates universe" >> /etc/apt/sources.list \
  && echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial multiverse" >> /etc/apt/sources.list \
  && echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial-updates multiverse" >> /etc/apt/sources.list \
  && echo "deb http://ch.archive.ubuntu.com/ubuntu/ xenial-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev libopencv-objdetect-dev libopencv-highgui-dev libopencv-legacy-dev libopencv-contrib-dev libopencv-videostab-dev libopencv-superres-dev libopencv-ocl-dev libcv-dev libhighgui-dev libcvaux-dev libgtk2.0-dev libglib2.0-dev libgdk-pixbuf2.0-dev libpango1.0-dev libatk1.0-dev libcairo2-dev libglib2.0-0 libglib2.0-bin libglib2.0-0
RUN pip install flask opencv-python

COPY entrypoint.sh /

EXPOSE 5000

ENTRYPOINT "/entrypoint.sh"
