FROM python:3.12-slim as slim

ENV WORKDIR=/usr/workspace
WORKDIR ${WORKDIR}

# Chromeの依存関係をインストール　 + タイムゾーンを日本時間に設定
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tzdata \
    && curl -sSL https://dl.google.com/linux/linux_signing_key.pub -o /usr/share/keyrings/google-linux-signing-key.asc \
    && echo "deb [signed-by=/usr/share/keyrings/google-linux-signing-key.asc arch=amd64] https://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get remove -y google-chrome-stable curl \
    && rm /usr/share/keyrings/google-linux-signing-key.asc \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

FROM slim as dev

COPY requirements.dev.txt ${WORKDIR}/requirements.txt
RUN pip install -U pip && \
    pip install --no-cache-dir -r ${WORKDIR}/requirements.txt

FROM slim as build

COPY requirements.deploy.txt ${WORKDIR}/requirements.txt
RUN pip install -U pip && \
    pip install --no-cache-dir -r ${WORKDIR}/requirements.txt
COPY . ${WORKDIR}

FROM slim as deploy

COPY --from=build ${WORKDIR} ${WORKDIR}
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

CMD ["python", "app.py"]
