// kairyu-f1a-mock is a deliberately tiny OpenAI-compatible backend for the
// 200-replica kind churn gate. It uses only the Go standard library and builds
// to one static binary so each replica consumes a few MiB instead of starting a
// Python runtime.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

const (
	maxRequestBytes = 1 << 20

	// The Pod manifest grants 15 seconds for preStop plus process shutdown.
	// Keep the drain command and HTTP server shutdown below that bound with one
	// second of scheduling margin: 9s + 5s < 15s.
	maxDrainCommandTimeout = 9 * time.Second
	serverShutdownTimeout  = 5 * time.Second
	drainRequestTimeout    = 1 * time.Second
	drainRetryInterval     = 100 * time.Millisecond
)

type identity struct {
	PodName string `json:"pod_name"`
	PodUID  string `json:"pod_uid"`
	PodIP   string `json:"pod_ip,omitempty"`
}

type server struct {
	identity      identity
	responseDelay time.Duration
	ready         atomic.Bool
	sequence      atomic.Uint64
}

type completionRequest struct {
	Model  string          `json:"model"`
	Prompt json.RawMessage `json:"prompt"`
	N      int             `json:"n"`
	Stream bool            `json:"stream"`
}

type chatRequest struct {
	Model    string `json:"model"`
	N        int    `json:"n"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
}

func newServer(id identity) *server {
	s := &server{identity: id}
	s.ready.Store(true)
	return s
}

func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.health)
	mux.HandleFunc("/readyz", s.readyz)
	mux.HandleFunc("/backends", s.backends)
	mux.HandleFunc("/v1/completions", s.completions)
	mux.HandleFunc("/v1/chat/completions", s.chatCompletions)
	mux.HandleFunc("/admin/drain", s.drain)
	return mux
}

func (s *server) responseID(prefix string) string {
	return fmt.Sprintf(
		"%s-%s-%d",
		prefix,
		safeID(s.identity.PodName),
		s.sequence.Add(1),
	)
}

func safeID(value string) string {
	value = strings.Map(func(char rune) rune {
		switch {
		case char >= 'a' && char <= 'z':
			return char
		case char >= 'A' && char <= 'Z':
			return char
		case char >= '0' && char <= '9':
			return char
		case char == '-', char == '_':
			return char
		default:
			return '-'
		}
	}, value)
	if value == "" {
		return "unknown"
	}
	return value
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	if err := json.NewEncoder(writer).Encode(value); err != nil {
		log.Printf("encode response: %v", err)
	}
}

func requireMethod(writer http.ResponseWriter, request *http.Request, method string) bool {
	if request.Method == method {
		return true
	}
	writer.Header().Set("Allow", method)
	writeJSON(writer, http.StatusMethodNotAllowed, map[string]any{
		"error": map[string]string{
			"message": "method not allowed",
			"type":    "invalid_request_error",
		},
	})
	return false
}

func decodeRequest(writer http.ResponseWriter, request *http.Request, value any) bool {
	decoder := json.NewDecoder(io.LimitReader(request.Body, maxRequestBytes))
	if err := decoder.Decode(value); err != nil {
		writeJSON(writer, http.StatusBadRequest, map[string]any{
			"error": map[string]string{
				"message": "invalid JSON request",
				"type":    "invalid_request_error",
			},
		})
		return false
	}
	return true
}

func normalizedN(n int) int {
	if n < 1 {
		return 1
	}
	if n > 16 {
		return 16
	}
	return n
}

func normalizedModel(model string) string {
	if model == "" {
		return "f1a"
	}
	return model
}

func (s *server) content() string {
	return fmt.Sprintf(
		"kairyu-f1a-mock pod_name=%s pod_uid=%s",
		s.identity.PodName,
		s.identity.PodUID,
	)
}

func (s *server) waitForResponse(request *http.Request) bool {
	return waitContext(request.Context(), s.responseDelay) == nil
}

func (s *server) health(writer http.ResponseWriter, request *http.Request) {
	if !requireMethod(writer, request, http.MethodGet) {
		return
	}
	writeJSON(writer, http.StatusOK, map[string]any{
		"status":   "ok",
		"pod_name": s.identity.PodName,
		"pod_uid":  s.identity.PodUID,
		"pod_ip":   s.identity.PodIP,
	})
}

func (s *server) readyz(writer http.ResponseWriter, request *http.Request) {
	if !requireMethod(writer, request, http.MethodGet) {
		return
	}
	status := http.StatusOK
	state := "ready"
	if !s.ready.Load() {
		status = http.StatusServiceUnavailable
		state = "draining"
	}
	writeJSON(writer, status, map[string]any{
		"status":   state,
		"pod_name": s.identity.PodName,
		"pod_uid":  s.identity.PodUID,
		"pod_ip":   s.identity.PodIP,
	})
}

func (s *server) backends(writer http.ResponseWriter, request *http.Request) {
	if !requireMethod(writer, request, http.MethodGet) {
		return
	}
	writeJSON(writer, http.StatusOK, map[string]any{
		"attention_backend": nil,
		"source":            "f1a-mock",
		"kernel_tier":       "mock",
		"role":              "engine-host",
		"versions":          map[string]string{"f1a-mock": "1"},
		"engines": []map[string]any{{
			"model":                "f1a",
			"engine_backend":       "mock",
			"attention_backend":    nil,
			"tensor_parallel_size": 1,
		}},
		"pod_name": s.identity.PodName,
		"pod_uid":  s.identity.PodUID,
		"pod_ip":   s.identity.PodIP,
	})
}

func (s *server) drain(writer http.ResponseWriter, request *http.Request) {
	if !requireMethod(writer, request, http.MethodPost) {
		return
	}
	s.ready.Store(false)
	writeJSON(writer, http.StatusOK, map[string]any{
		"status":   "draining",
		"pod_name": s.identity.PodName,
		"pod_uid":  s.identity.PodUID,
		"pod_ip":   s.identity.PodIP,
	})
}

func (s *server) completions(writer http.ResponseWriter, request *http.Request) {
	if !requireMethod(writer, request, http.MethodPost) {
		return
	}
	var input completionRequest
	if !decodeRequest(writer, request, &input) {
		return
	}
	if !s.waitForResponse(request) {
		return
	}
	n := normalizedN(input.N)
	model := normalizedModel(input.Model)
	id := s.responseID("cmpl")
	created := time.Now().Unix()
	text := s.content()
	if input.Stream {
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		writer.Header().Set("X-Accel-Buffering", "no")
		flusher, _ := writer.(http.Flusher)
		for index := 0; index < n; index++ {
			writeSSE(writer, map[string]any{
				"id":      id,
				"object":  "text_completion",
				"created": created,
				"model":   model,
				"choices": []map[string]any{{
					"index":         index,
					"text":          text,
					"finish_reason": "stop",
				}},
				"pod_name": s.identity.PodName,
				"pod_uid":  s.identity.PodUID,
			})
		}
		fmt.Fprint(writer, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	choices := make([]map[string]any, n)
	for index := range choices {
		choices[index] = map[string]any{
			"index":         index,
			"text":          text,
			"logprobs":      nil,
			"finish_reason": "stop",
		}
	}
	writeJSON(writer, http.StatusOK, map[string]any{
		"id":       id,
		"object":   "text_completion",
		"created":  created,
		"model":    model,
		"choices":  choices,
		"usage":    usage(n),
		"pod_name": s.identity.PodName,
		"pod_uid":  s.identity.PodUID,
	})
}

func (s *server) chatCompletions(writer http.ResponseWriter, request *http.Request) {
	if !requireMethod(writer, request, http.MethodPost) {
		return
	}
	var input chatRequest
	if !decodeRequest(writer, request, &input) {
		return
	}
	if !s.waitForResponse(request) {
		return
	}
	n := normalizedN(input.N)
	model := normalizedModel(input.Model)
	id := s.responseID("chatcmpl")
	created := time.Now().Unix()
	content := s.content()
	if input.Stream {
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		writer.Header().Set("X-Accel-Buffering", "no")
		flusher, _ := writer.(http.Flusher)
		for index := 0; index < n; index++ {
			writeSSE(writer, map[string]any{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]any{{
					"index": index,
					"delta": map[string]string{
						"role":    "assistant",
						"content": content,
					},
					"finish_reason": "stop",
				}},
				"pod_name": s.identity.PodName,
				"pod_uid":  s.identity.PodUID,
			})
		}
		fmt.Fprint(writer, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	choices := make([]map[string]any, n)
	for index := range choices {
		choices[index] = map[string]any{
			"index": index,
			"message": map[string]string{
				"role":    "assistant",
				"content": content,
			},
			"logprobs":      nil,
			"finish_reason": "stop",
		}
	}
	writeJSON(writer, http.StatusOK, map[string]any{
		"id":       id,
		"object":   "chat.completion",
		"created":  created,
		"model":    model,
		"choices":  choices,
		"usage":    usage(n),
		"pod_name": s.identity.PodName,
		"pod_uid":  s.identity.PodUID,
	})
}

func usage(n int) map[string]int {
	return map[string]int{
		"prompt_tokens":     1,
		"completion_tokens": n,
		"total_tokens":      n + 1,
	}
}

func writeSSE(writer io.Writer, value any) {
	payload, err := json.Marshal(value)
	if err != nil {
		log.Printf("encode SSE response: %v", err)
		return
	}
	fmt.Fprintf(writer, "data: %s\n\n", payload)
}

type gracefulHTTPServer interface {
	ListenAndServe() error
	Shutdown(context.Context) error
}

func serveUntilStopped(
	httpServer gracefulHTTPServer,
	markDraining func(),
	stop <-chan os.Signal,
	shutdownTimeout time.Duration,
) error {
	serveDone := make(chan error, 1)
	go func() {
		serveDone <- httpServer.ListenAndServe()
	}()

	select {
	case err := <-serveDone:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-stop:
		markDraining()
		shutdownContext, shutdownCancel := context.WithTimeout(
			context.Background(),
			shutdownTimeout,
		)
		defer shutdownCancel()

		shutdownError := httpServer.Shutdown(shutdownContext)
		// Shutdown closes the listener before it finishes waiting for active
		// handlers. Always join ListenAndServe after Shutdown itself returns so
		// main cannot exit while an in-flight response is still being drained.
		serveError := <-serveDone
		if shutdownError != nil {
			if serveError != nil && !errors.Is(serveError, http.ErrServerClosed) {
				return errors.Join(shutdownError, serveError)
			}
			return shutdownError
		}
		if serveError != nil && !errors.Is(serveError, http.ErrServerClosed) {
			return serveError
		}
		return nil
	}
}

func runServer() error {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	if _, err := strconv.ParseUint(port, 10, 16); err != nil {
		return fmt.Errorf("invalid PORT %q: %w", port, err)
	}
	responseDelay := time.Duration(0)
	if raw := os.Getenv("F1A_MOCK_RESPONSE_DELAY"); raw != "" {
		parsed, err := time.ParseDuration(raw)
		if err != nil || parsed < 0 || parsed >= serverShutdownTimeout {
			return fmt.Errorf(
				"invalid F1A_MOCK_RESPONSE_DELAY %q: require 0 <= delay < %s",
				raw,
				serverShutdownTimeout,
			)
		}
		responseDelay = parsed
	}
	id := identity{
		PodName: os.Getenv("POD_NAME"),
		PodUID:  os.Getenv("POD_UID"),
		PodIP:   os.Getenv("POD_IP"),
	}
	if id.PodName == "" {
		id.PodName = "local"
	}
	if id.PodUID == "" {
		id.PodUID = "local"
	}

	mock := newServer(id)
	mock.responseDelay = responseDelay
	httpServer := &http.Server{
		Addr:              ":" + port,
		Handler:           mock.routes(),
		ReadHeaderTimeout: 2 * time.Second,
		ReadTimeout:       5 * time.Second,
		WriteTimeout:      5 * time.Second,
		IdleTimeout:       30 * time.Second,
	}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(stop)

	log.Printf(
		"kairyu F1a mock listening on %s pod_name=%s pod_uid=%s",
		httpServer.Addr,
		id.PodName,
		id.PodUID,
	)
	if err := serveUntilStopped(
		httpServer,
		func() { mock.ready.Store(false) },
		stop,
		serverShutdownTimeout,
	); err != nil {
		return fmt.Errorf("serve or graceful shutdown: %w", err)
	}
	return nil
}

func waitContext(ctx context.Context, duration time.Duration) error {
	if duration == 0 {
		return nil
	}
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func acknowledgeDrain(
	ctx context.Context,
	client *http.Client,
	url string,
) error {
	var lastError error
	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			if lastError == nil {
				lastError = err
			}
			return fmt.Errorf(
				"drain was not acknowledged before deadline after %d attempts: %w",
				attempt-1,
				lastError,
			)
		}

		request, err := http.NewRequestWithContext(
			ctx,
			http.MethodPost,
			url,
			nil,
		)
		if err != nil {
			return fmt.Errorf("build drain request: %w", err)
		}
		response, err := client.Do(request)
		if err == nil {
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()
			if response.StatusCode >= 200 && response.StatusCode < 300 {
				return nil
			}
			err = fmt.Errorf("drain returned HTTP %d", response.StatusCode)
		}
		lastError = err
		if err := waitContext(ctx, drainRetryInterval); err != nil {
			return fmt.Errorf(
				"drain was not acknowledged before deadline after %d attempts: %w",
				attempt,
				lastError,
			)
		}
	}
}

func performDrain(
	ctx context.Context,
	client *http.Client,
	url string,
	grace time.Duration,
	waitGrace func(context.Context, time.Duration) error,
) error {
	deadline, hasDeadline := ctx.Deadline()
	if !hasDeadline {
		return fmt.Errorf("drain command requires an explicit deadline")
	}
	ackDeadline := deadline.Add(-grace)
	if !ackDeadline.After(time.Now()) {
		return fmt.Errorf("drain grace %s leaves no time for acknowledgement", grace)
	}
	ackContext, ackCancel := context.WithDeadline(ctx, ackDeadline)
	defer ackCancel()
	if err := acknowledgeDrain(ackContext, client, url); err != nil {
		// Do not consume the post-ACK propagation grace when no ACK occurred.
		return err
	}

	log.Printf("drain acknowledged; waiting %s", grace)
	if err := waitGrace(ctx, grace); err != nil {
		return fmt.Errorf("post-drain grace interrupted: %w", err)
	}
	return nil
}

func runDrain(arguments []string) error {
	flags := flag.NewFlagSet("drain", flag.ContinueOnError)
	url := flags.String("url", "http://127.0.0.1:8080/admin/drain", "drain URL")
	grace := flags.Duration("grace", 5*time.Second, "post-drain grace period")
	timeout := flags.Duration(
		"timeout",
		maxDrainCommandTimeout,
		"total drain command deadline",
	)
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *grace < 0 {
		return fmt.Errorf("grace must be non-negative")
	}
	if *timeout <= 0 || *timeout > maxDrainCommandTimeout {
		return fmt.Errorf(
			"timeout must be > 0 and <= %s, got %s",
			maxDrainCommandTimeout,
			*timeout,
		)
	}
	if *grace >= *timeout {
		return fmt.Errorf(
			"grace must be less than timeout, got grace=%s timeout=%s",
			*grace,
			*timeout,
		)
	}

	commandContext, commandCancel := context.WithTimeout(
		context.Background(),
		*timeout,
	)
	defer commandCancel()
	requestTimeout := drainRequestTimeout
	if acknowledgementBudget := *timeout - *grace; acknowledgementBudget < requestTimeout {
		requestTimeout = acknowledgementBudget
	}
	client := &http.Client{Timeout: requestTimeout}
	return performDrain(
		commandContext,
		client,
		*url,
		*grace,
		waitContext,
	)
}

func main() {
	var err error
	if len(os.Args) > 1 && os.Args[1] == "drain" {
		err = runDrain(os.Args[2:])
	} else {
		err = runServer()
	}
	if err != nil {
		log.Fatal(err)
	}
}
