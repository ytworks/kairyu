package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
)

func testServer(t *testing.T) (*server, *httptest.Server) {
	t.Helper()
	mock := newServer(identity{
		PodName: "f1a-replica-017",
		PodUID:  "uid-017",
		PodIP:   "10.0.0.17",
	})
	httpServer := httptest.NewServer(mock.routes())
	t.Cleanup(httpServer.Close)
	return mock, httpServer
}

func getJSON(t *testing.T, url string, expectedStatus int) map[string]any {
	t.Helper()
	response, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != expectedStatus {
		t.Fatalf("GET %s returned %d, want %d", url, response.StatusCode, expectedStatus)
	}
	var body map[string]any
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	return body
}

func postJSON(
	t *testing.T,
	url string,
	body string,
	expectedStatus int,
) (http.Header, map[string]any) {
	t.Helper()
	response, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != expectedStatus {
		payload, _ := io.ReadAll(response.Body)
		t.Fatalf(
			"POST %s returned %d, want %d: %s",
			url,
			response.StatusCode,
			expectedStatus,
			payload,
		)
	}
	var payload map[string]any
	if err := json.NewDecoder(response.Body).Decode(&payload); err != nil {
		t.Fatal(err)
	}
	return response.Header, payload
}

func TestHealthReadyAndDrain(t *testing.T) {
	mock, httpServer := testServer(t)

	health := getJSON(t, httpServer.URL+"/health", http.StatusOK)
	if health["pod_name"] != "f1a-replica-017" || health["pod_uid"] != "uid-017" {
		t.Fatalf("health identity = %#v", health)
	}
	getJSON(t, httpServer.URL+"/readyz", http.StatusOK)

	_, drained := postJSON(
		t,
		httpServer.URL+"/admin/drain",
		"{}",
		http.StatusOK,
	)
	if drained["status"] != "draining" || mock.ready.Load() {
		t.Fatalf("drain response = %#v, ready=%v", drained, mock.ready.Load())
	}
	ready := getJSON(t, httpServer.URL+"/readyz", http.StatusServiceUnavailable)
	if ready["status"] != "draining" {
		t.Fatalf("ready response = %#v", ready)
	}
	// Draining removes the endpoint from discovery but deliberately does not
	// reject a request selected just before the readiness transition.
	_, chat := postJSON(
		t,
		httpServer.URL+"/v1/chat/completions",
		`{"model":"f1a","messages":[{"role":"user","content":"hello"}]}`,
		http.StatusOK,
	)
	if chat["pod_uid"] != "uid-017" {
		t.Fatalf("chat identity = %#v", chat)
	}
}

func TestChatCompletionIncludesReplicaIdentityAndNChoices(t *testing.T) {
	_, httpServer := testServer(t)
	headers, body := postJSON(
		t,
		httpServer.URL+"/v1/chat/completions",
		`{"model":"f1a","n":2,"messages":[{"role":"user","content":"hello"}]}`,
		http.StatusOK,
	)
	if headers.Get("Content-Type") != "application/json" {
		t.Fatalf("content type = %q", headers.Get("Content-Type"))
	}
	if body["pod_name"] != "f1a-replica-017" || body["pod_uid"] != "uid-017" {
		t.Fatalf("chat identity = %#v", body)
	}
	choices, ok := body["choices"].([]any)
	if !ok || len(choices) != 2 {
		t.Fatalf("chat choices = %#v", body["choices"])
	}
	first := choices[0].(map[string]any)
	message := first["message"].(map[string]any)
	content := message["content"].(string)
	if !strings.Contains(content, "pod_name=f1a-replica-017") ||
		!strings.Contains(content, "pod_uid=uid-017") {
		t.Fatalf("chat content = %q", content)
	}
}

func TestCompletionAndStreamingChat(t *testing.T) {
	_, httpServer := testServer(t)
	_, completion := postJSON(
		t,
		httpServer.URL+"/v1/completions",
		`{"model":"f1a","prompt":"hello"}`,
		http.StatusOK,
	)
	if completion["object"] != "text_completion" ||
		completion["pod_uid"] != "uid-017" {
		t.Fatalf("completion = %#v", completion)
	}

	response, err := http.Post(
		httpServer.URL+"/v1/chat/completions",
		"application/json",
		strings.NewReader(
			`{"model":"f1a","stream":true,"messages":[{"role":"user","content":"hello"}]}`,
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK ||
		!strings.HasPrefix(response.Header.Get("Content-Type"), "text/event-stream") ||
		!strings.Contains(string(payload), `"pod_uid":"uid-017"`) ||
		!strings.HasSuffix(string(payload), "data: [DONE]\n\n") {
		t.Fatalf(
			"stream status=%d content-type=%q payload=%q",
			response.StatusCode,
			response.Header.Get("Content-Type"),
			payload,
		)
	}
}

func TestBackendsAndMethodValidation(t *testing.T) {
	_, httpServer := testServer(t)
	backends := getJSON(t, httpServer.URL+"/backends", http.StatusOK)
	if backends["source"] != "f1a-mock" || backends["pod_uid"] != "uid-017" {
		t.Fatalf("backends = %#v", backends)
	}

	response, err := http.Get(httpServer.URL + "/admin/drain")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusMethodNotAllowed ||
		response.Header.Get("Allow") != http.MethodPost {
		t.Fatalf(
			"GET drain status=%d allow=%q",
			response.StatusCode,
			response.Header.Get("Allow"),
		)
	}
}

type blockingShutdownServer struct {
	listenStarted   chan struct{}
	shutdownStarted chan struct{}
	releaseShutdown chan struct{}
}

func newBlockingShutdownServer() *blockingShutdownServer {
	return &blockingShutdownServer{
		listenStarted:   make(chan struct{}),
		shutdownStarted: make(chan struct{}),
		releaseShutdown: make(chan struct{}),
	}
}

func (s *blockingShutdownServer) ListenAndServe() error {
	close(s.listenStarted)
	<-s.shutdownStarted
	return http.ErrServerClosed
}

func (s *blockingShutdownServer) Shutdown(ctx context.Context) error {
	close(s.shutdownStarted)
	select {
	case <-s.releaseShutdown:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestServeUntilStoppedWaitsForGracefulShutdown(t *testing.T) {
	httpServer := newBlockingShutdownServer()
	stop := make(chan os.Signal, 1)
	draining := make(chan struct{})
	result := make(chan error, 1)
	go func() {
		result <- serveUntilStopped(
			httpServer,
			func() { close(draining) },
			stop,
			time.Second,
		)
	}()

	select {
	case <-httpServer.listenStarted:
	case <-time.After(time.Second):
		t.Fatal("server did not start listening")
	}
	stop <- syscall.SIGTERM
	select {
	case <-httpServer.shutdownStarted:
	case <-time.After(time.Second):
		t.Fatal("graceful shutdown did not start")
	}
	select {
	case <-draining:
	default:
		t.Fatal("server was not marked draining before shutdown")
	}
	select {
	case err := <-result:
		t.Fatalf("serve returned before graceful shutdown completed: %v", err)
	default:
	}

	close(httpServer.releaseShutdown)
	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("serve returned an error after graceful shutdown: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("serve did not join graceful shutdown")
	}
}

func TestPerformDrainWaitsGraceOnlyAfterAcknowledgement(t *testing.T) {
	requests := 0
	httpServer := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			requests++
			writer.WriteHeader(http.StatusNoContent)
		},
	))
	defer httpServer.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	waited := false
	grace := 25 * time.Millisecond
	err := performDrain(
		ctx,
		httpServer.Client(),
		httpServer.URL,
		grace,
		func(ctx context.Context, duration time.Duration) error {
			waited = true
			if duration != grace {
				t.Fatalf("grace = %s, want %s", duration, grace)
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("performDrain returned an error: %v", err)
	}
	if requests != 1 {
		t.Fatalf("drain requests = %d, want 1", requests)
	}
	if !waited {
		t.Fatal("post-ACK grace was not observed")
	}
}

func TestPerformDrainFailureDoesNotWaitPostAckGrace(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	httpServer := httptest.NewServer(http.HandlerFunc(
		func(writer http.ResponseWriter, request *http.Request) {
			http.Error(writer, "not ready", http.StatusServiceUnavailable)
			cancel()
		},
	))
	defer httpServer.Close()

	waited := false
	err := performDrain(
		ctx,
		httpServer.Client(),
		httpServer.URL,
		20*time.Millisecond,
		func(ctx context.Context, duration time.Duration) error {
			waited = true
			return nil
		},
	)
	if err == nil || !strings.Contains(err.Error(), "not acknowledged before deadline") {
		t.Fatalf("performDrain error = %v", err)
	}
	if waited {
		t.Fatal("post-ACK grace ran even though drain was not acknowledged")
	}
}

func TestRunDrainRejectsDeadlineOutsidePodBudget(t *testing.T) {
	err := runDrain([]string{"-timeout=10s", "-grace=5s"})
	if err == nil || !strings.Contains(err.Error(), "timeout must be > 0 and <= 9s") {
		t.Fatalf("runDrain error = %v", err)
	}
}
