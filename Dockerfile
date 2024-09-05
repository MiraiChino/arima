FROM python:3.12 as dev

ENV WORKDIR=/usr/workspace
WORKDIR ${WORKDIR}

# Chromeの依存関係をインストール
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] https://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get remove -y google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ${WORKDIR}/requirements.txt
RUN pip install -U pip && \
    pip install --no-cache-dir -r ${WORKDIR}/requirements.txt

FROM dev as build
COPY . ${WORKDIR}

FROM build as deploy
CMD ["python", "app.py"]
