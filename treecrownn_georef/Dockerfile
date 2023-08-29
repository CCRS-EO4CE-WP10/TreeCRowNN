FROM python:3.10

WORKDIR /TreeCRowNN/treecrownn_georef
COPY ./  .
RUN make install
CMD ["python","./treecrownn_georef/main.py"]