# hadolint global ignore=SC1090,SC1091

#Archipelago
FROM python:3.12-slim AS archipelago-base
ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY . .

RUN apt-get update; \
    apt-get install -y --no-install-recommends \
    git=1:2.39.5-0+deb12u1 \
    gcc=4:12.2.0-3 \
    libc6-dev=2.36-9+deb12u9 \
    libtk8.6=8.6.13-2 \
    g++=4:12.2.0-3; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

#create and activate venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="/opt/venv/bin:$PATH"

RUN $VIRTUAL_ENV/bin/pip install -r requirements.txt
RUN $VIRTUAL_ENV/bin/python ModuleUpdate.py -y
RUN $VIRTUAL_ENV/bin/cythonize -i _speedups.pyx

RUN ln -s -T /data/host.yaml /app/host.yaml

RUN apt-get purge \
    git \
    gcc \
    libc6-dev \
    g++; \
    apt-get autoremove

FROM archipelago-base AS archipelago-multiserver
VOLUME /data
WORKDIR /data
EXPOSE 38281
ENTRYPOINT [ "/opt/venv/bin/python", "/app/MultiServer.py", "--port", "38281" ]

ARG ARCHITECTURE=$(uname -m)

#Source
FROM scratch AS release
WORKDIR /release
#Not sure how to build this project. Grab release instead.
ADD https://github.com/Ijwu/Enemizer/releases/latest/download/ubuntu.16.04-x64.zip Enemizer.zip
#Enemizer
FROM alpine:3.21 AS enemizer
WORKDIR /release
COPY --from=release /release/Enemizer.zip .
#No release for arm architecture. Skip.
RUN if [ "$ARCHITECTURE" = "x86_64" ]; then \
    apk add unzip=6.0-r15 --no-cache; \
    unzip -u Enemizer.zip -d EnemizerCLI; \
    chmod -R 777 EnemizerCLI; \
    else touch EnemizerCLI; fi


FROM archipelago-base AS archipelago-generate
# Copy necessary components
COPY --from=enemizer /release/EnemizerCLI /tmp/EnemizerCLI

#No release for arm architecture. Skip.
RUN if [ "$ARCHITECTURE" = "x86_64" ]; then \
    cp /tmp/EnemizerCLI EnemizerCLI; \
    fi; \
    rm -rf /tmp/EnemizerCLI

VOLUME /data
WORKDIR /data
ENTRYPOINT [ "/opt/venv/bin/python", "/app/Generate.py" ]
CMD [ "--player_files_path", "/data/Players", "--outputpath", "/data/output" ]
