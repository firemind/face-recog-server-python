from tensorflow/tensorflow

RUN pip install flask 
RUN pip install requests

COPY entrypoint.sh /

EXPOSE 5000

ENTRYPOINT "/entrypoint.sh"
