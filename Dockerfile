from tensorflow/tensorflow

RUN pip install Flask

COPY entrypoint.sh /

ENTRYPOINT "/entrypoint.sh"
