FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /jange_street_kaggle

# Add setup.py to install dependecies and cache them
COPY setup.py /jange_street_kaggle
COPY README.md /jange_street_kaggle
COPY jange_street_kaggle/bin/jange_street_kaggle /jange_street_kaggle/jange_street_kaggle/bin/jange_street_kaggle
COPY jange_street_kaggle/version.py /jange_street_kaggle/jange_street_kaggle/version.py
RUN pip install -e /jange_street_kaggle

# Add source code
# Everything after this line is uncached
COPY . /jange_street_kaggle

RUN pip install -e /jange_street_kaggle

ENTRYPOINT ["jange_street_kaggle"]

CMD ["--help"]