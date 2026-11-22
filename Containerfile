# Un solo Containerfile para ambos binarios: --build-arg CMD=nodo | coordinador
FROM docker.io/library/golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
ARG CMD
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/app ./cmd/${CMD}

FROM docker.io/library/alpine:3.20
COPY --from=build /out/app /app
ENTRYPOINT ["/app"]
