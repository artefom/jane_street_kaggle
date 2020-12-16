FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /jane_street_kaggle

# Add setup.py to install dependecies and cache them
COPY setup.py /jane_street_kaggle
COPY README.md /jane_street_kaggle
COPY jane_street_kaggle/bin/jane_street_kaggle /jane_street_kaggle/jane_street_kaggle/bin/jane_street_kaggle
COPY jane_street_kaggle/version.py /jane_street_kaggle/jane_street_kaggle/version.py
RUN pip install -e /jane_street_kaggle

# Add source code
# Everything after this line is uncached
COPY . /jane_street_kaggle

RUN pip install -e /jane_street_kaggle

ENTRYPOINT ["jane_street_kaggle"]

CMD ["--help"]