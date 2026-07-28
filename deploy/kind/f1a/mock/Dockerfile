FROM golang:1.24-bookworm@sha256:1a6d4452c65dea36aac2e2d606b01b4a029ec90cc1ae53890540ce6173ea77ac AS builder

WORKDIR /src
COPY go.mod main.go main_test.go ./
RUN go test ./...
RUN CGO_ENABLED=0 GOOS=linux go build \
    -trimpath \
    -ldflags="-s -w -buildid=" \
    -o /out/kairyu-f1a-mock \
    .

FROM scratch
COPY --from=builder /out/kairyu-f1a-mock /kairyu-f1a-mock
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/kairyu-f1a-mock"]
