FROM public.ecr.aws/docker/library/r-base:4.4.2 AS base

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && apt-get upgrade -y \
    && apt-get install -y libpq-dev curl libfontconfig1-dev libharfbuzz-dev \
    libfribidi-dev libfreetype6-dev libcurl4-gnutls-dev \
    libxml2-dev libtiff-dev libx11-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base AS compilation

WORKDIR /fpe

RUN R -e "install.packages('renv', repos = c(CRAN = 'https://cloud.r-project.org'))"

RUN mkdir -p renv

COPY renv.lock renv.lock
COPY .Rprofile .Rprofile
COPY renv/activate.R renv/activate.R
COPY renv/settings.json renv/settings.json
RUN mkdir renv/.cache
ENV RENV_PATHS_CACHE=renv/.cache

RUN R -e "renv::restore()"


FROM base AS run

WORKDIR /fpe
COPY --from=compilation /fpe .

COPY rank-dataset.R .
COPY config.yml .

RUN mkdir rank-dataset-output

ENTRYPOINT ["Rscript"]