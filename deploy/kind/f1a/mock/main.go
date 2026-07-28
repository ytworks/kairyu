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

const maxRequestBytes = 1 << 20

type identity struct {
	PodName string `json:"pod_name"`
	PodUID  string `json:"pod_uid"`
	PodIP   string `json:"pod_ip,omitempty"`
}

type server struct {
	identity identity
	ready    atomic.Bool
	sequence atomic.Uint64
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

func runServer() error {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	if _, err := strconv.ParseUint(port, 10, 16); err != nil {
		return fmt.Errorf("invalid PORT %q: %w", port, err)
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
	go func() {
		<-stop
		mock.ready.Store(false)
		shutdownContext, shutdownCancel := context.WithTimeout(
			context.Background(),
			5*time.Second,
		)
		defer shutdownCancel()
		if err := httpServer.Shutdown(shutdownContext); err != nil {
			log.Printf("graceful shutdown: %v", err)
		}
	}()

	log.Printf(
		"kairyu F1a mock listening on %s pod_name=%s pod_uid=%s",
		httpServer.Addr,
		id.PodName,
		id.PodUID,
	)
	err := httpServer.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func runDrain(arguments []string) error {
	flags := flag.NewFlagSet("drain", flag.ContinueOnError)
	url := flags.String("url", "http://127.0.0.1:8080/admin/drain", "drain URL")
	grace := flags.Duration("grace", 5*time.Second, "post-drain grace period")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *grace < 0 {
		return fmt.Errorf("grace must be non-negative")
	}

	var lastError error
	client := &http.Client{Timeout: time.Second}
	for attempt := 0; attempt < 20; attempt++ {
		request, err := http.NewRequest(http.MethodPost, *url, nil)
		if err != nil {
			return err
		}
		response, err := client.Do(request)
		if err == nil {
			_, _ = io.Copy(io.Discard, response.Body)
			_ = response.Body.Close()
			if response.StatusCode >= 200 && response.StatusCode < 300 {
				log.Printf("drain acknowledged; waiting %s", *grace)
				time.Sleep(*grace)
				return nil
			}
			err = fmt.Errorf("drain returned HTTP %d", response.StatusCode)
		}
		lastError = err
		time.Sleep(100 * time.Millisecond)
	}
	time.Sleep(*grace)
	return fmt.Errorf("drain was not acknowledged: %w", lastError)
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
