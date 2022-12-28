FROM joyzoursky/python-chromedriver:3.9-selenium as dev

ENV WORKDIR=/usr/workspace
WORKDIR ${WORKDIR}

COPY requirements.txt ${WORKDIR}
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

FROM dev as deploy
COPY . ${WORKDIR}
CMD ["python", "app.py"]
