package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/coordinator"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/dataset"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/store"
)

type Config struct {
	HTTPAddr     string
	DataPath     string
	Workers      int
	TestFraction float64
	DefaultTrees int
	SwaggerUI    string
	OpenAPI      []byte
}

type Server struct {
	cfg   Config
	hub   *coordinator.Hub
	store *store.Store

	mu   sync.RWMutex
	data *dataset.Prepared

	trainMu sync.Mutex
}

func New(hub *coordinator.Hub, st *store.Store, cfg Config) *Server {
	if cfg.DefaultTrees <= 0 {
		cfg.DefaultTrees = 30
	}
	if cfg.TestFraction <= 0 {
		cfg.TestFraction = 0.1
	}
	return &Server{cfg: cfg, hub: hub, store: st}
}

func (s *Server) LoadData(workers int) error {
	if workers <= 0 {
		workers = s.cfg.Workers
	}
	recs, st, err := dataset.LoadCSV(s.cfg.DataPath, workers)
	if err != nil {
		return err
	}
	data := dataset.Prepare(recs, st, s.cfg.TestFraction, workers)
	s.mu.Lock()
	s.data = data
	s.mu.Unlock()
	log.Printf("[api] dataset listo: %d filas, %d clases, %d distritos",
		len(data.Records), len(data.Classes), len(data.Districts))
	return nil
}

func (s *Server) prepared() *dataset.Prepared {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.data
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /api/v1/nodes", s.handleNodes)
	mux.HandleFunc("GET /api/v1/classes", s.handleClasses)
	mux.HandleFunc("POST /api/v1/train", s.handleTrain)
	mux.HandleFunc("POST /api/v1/predict", s.handlePredict)
	mux.HandleFunc("GET /api/v1/predictions", s.handleListPredictions)
	mux.HandleFunc("GET /api/v1/train-runs", s.handleListTrainRuns)
	mux.HandleFunc("GET /openapi.yaml", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
		_, _ = w.Write(s.cfg.OpenAPI)
	})
	mux.HandleFunc("GET /swagger/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(s.cfg.SwaggerUI))
	})
	mux.HandleFunc("GET /swagger", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/swagger/", http.StatusFound)
	})
	return withCORS(withTiming(mux))
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func withTiming(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &statusWriter{ResponseWriter: w, code: 200}
		next.ServeHTTP(rw, r)
		log.Printf("[http] %s %s -> %d (%.2f ms)", r.Method, r.URL.Path, rw.code,
			float64(time.Since(start).Microseconds())/1000)
	})
}

type statusWriter struct {
	http.ResponseWriter
	code int
}

func (w *statusWriter) WriteHeader(code int) {
	w.code = code
	w.ResponseWriter.WriteHeader(code)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	data := s.prepared()
	classes := 0
	if data != nil {
		classes = len(data.Classes)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status":     "ok",
		"nodes":      s.hub.NodeCount(),
		"classes":    classes,
		"data_ready": data != nil,
		"time":       time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) handleNodes(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"count": s.hub.NodeCount(),
		"nodes": s.hub.Status(),
	})
}

func (s *Server) handleClasses(w http.ResponseWriter, r *http.Request) {
	data := s.prepared()
	if data == nil {
		writeErr(w, http.StatusServiceUnavailable, "dataset aún no cargado")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"classes": data.Classes})
}

type trainRequest struct {
	Trees          int     `json:"trees"`
	MaxDepth       int     `json:"max_depth"`
	MinSamplesLeaf int     `json:"min_samples_leaf"`
	TestFraction   float64 `json:"test_fraction"`
}

func (s *Server) handleTrain(w http.ResponseWriter, r *http.Request) {
	if !s.trainMu.TryLock() {
		writeErr(w, http.StatusConflict, "ya hay un entrenamiento en curso")
		return
	}
	defer s.trainMu.Unlock()

	var req trainRequest
	if r.Body != nil && r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, http.StatusBadRequest, "JSON inválido: "+err.Error())
			return
		}
	}
	if req.Trees <= 0 {
		req.Trees = s.cfg.DefaultTrees
	}
	tf := s.cfg.TestFraction
	if req.TestFraction > 0 {
		tf = req.TestFraction
	}

	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Minute)
	defer cancel()

	begin := time.Now()
	res, err := s.hub.Train(ctx, coordinator.TrainConfig{
		Params: rf.Params{
			NumTrees:       req.Trees,
			MaxDepth:       req.MaxDepth,
			MinSamplesLeaf: req.MinSamplesLeaf,
		},
		DataPath:     s.cfg.DataPath,
		TestFraction: tf,
	})
	elapsed := time.Since(begin)

	run := &store.TrainRun{
		TreesTotal: req.Trees,
		Nodes:      len(res),
		DurationMs: float64(elapsed.Microseconds()) / 1000,
		DataPath:   s.cfg.DataPath,
		CreatedAt:  time.Now().UTC(),
	}
	for _, n := range res {
		tr := store.TrainNodeResult{
			NodeID: n.NodeID, Shard: n.Shard, Rows: n.Rows, Trees: n.Trees, Millis: n.Millis,
		}
		if n.Err != nil {
			tr.Error = n.Err.Error()
		}
		run.Results = append(run.Results, tr)
	}
	_ = s.store.SaveTrainRun(context.Background(), run)

	code := http.StatusOK
	if err != nil {
		code = http.StatusPartialContent
	}
	writeJSON(w, code, map[string]any{
		"ok":          err == nil,
		"error":       errString(err),
		"duration_ms": run.DurationMs,
		"train_run":   run,
	})
}

type predictRequest struct {
	Ubigeo int `json:"ubigeo"`
	Anio   int `json:"anio"`
	Mes    int `json:"mes"`
}

type predictResponse struct {
	Ubigeo     int                `json:"ubigeo"`
	Anio       int                `json:"anio"`
	Mes        int                `json:"mes"`
	TopClass   string             `json:"top_class"`
	TopProb    float64            `json:"top_prob"`
	Probs      map[string]float64 `json:"probs"`
	LatencyMs  float64            `json:"latency_ms"`
	Nodes      int                `json:"nodes"`
	Trees      int                `json:"trees"`
	Cached     bool               `json:"cached"`
	Under100ms bool               `json:"under_100ms"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	var req predictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "JSON inválido: "+err.Error())
		return
	}
	if req.Ubigeo <= 0 || req.Anio < 2000 || req.Mes < 1 || req.Mes > 12 {
		writeErr(w, http.StatusBadRequest, "requiere ubigeo, anio (>=2000) y mes (1-12)")
		return
	}

	ctx := r.Context()
	if cached, ok := s.store.GetCachedPrediction(ctx, req.Ubigeo, req.Anio, req.Mes); ok {
		writeJSON(w, http.StatusOK, toPredictResponse(cached, true))
		return
	}

	data := s.prepared()
	if data == nil {
		writeErr(w, http.StatusServiceUnavailable, "dataset aún no cargado")
		return
	}
	x, ok := data.Features(req.Ubigeo, req.Anio, req.Mes, nil)
	if !ok {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("ubigeo %d no está en el dataset", req.Ubigeo))
		return
	}

	pctx, cancel := context.WithTimeout(ctx, coordinator.DefaultPredictTimeout)
	defer cancel()
	begin := time.Now()
	res, err := s.hub.Predict(pctx, x)
	latency := float64(time.Since(begin).Microseconds()) / 1000
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}

	probs := map[string]float64{}
	topClass := ""
	topProb := -1.0
	for i, name := range data.Classes {
		if i >= len(res.Probs) {
			break
		}
		p := float64(res.Probs[i])
		probs[name] = p
		if p > topProb {
			topProb = p
			topClass = name
		}
	}

	rec := &store.Prediction{
		Ubigeo: req.Ubigeo, Anio: req.Anio, Mes: req.Mes,
		TopClass: topClass, Probs: probs,
		LatencyMs: latency, Nodes: res.Nodes, Trees: res.Trees,
		CreatedAt: time.Now().UTC(),
	}
	_ = s.store.SavePrediction(context.Background(), rec)
	_ = s.store.CachePrediction(context.Background(), rec)
	_, _ = s.store.IncrPredictCounter(context.Background())

	writeJSON(w, http.StatusOK, toPredictResponse(rec, false))
}

func toPredictResponse(p *store.Prediction, fromCache bool) predictResponse {
	topProb := 0.0
	if p.Probs != nil {
		if v, ok := p.Probs[p.TopClass]; ok {
			topProb = v
		}
	}
	lat := p.LatencyMs
	if fromCache {
		lat = 0
	}
	return predictResponse{
		Ubigeo: p.Ubigeo, Anio: p.Anio, Mes: p.Mes,
		TopClass: p.TopClass, TopProb: topProb, Probs: p.Probs,
		LatencyMs: lat, Nodes: p.Nodes, Trees: p.Trees,
		Cached: fromCache || p.Cached, Under100ms: p.LatencyMs < 100 || fromCache,
	}
}

func (s *Server) handleListPredictions(w http.ResponseWriter, r *http.Request) {
	list, err := s.store.ListPredictions(r.Context(), 50)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if list == nil {
		list = []store.Prediction{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": list})
}

func (s *Server) handleListTrainRuns(w http.ResponseWriter, r *http.Request) {
	list, err := s.store.ListTrainRuns(r.Context(), 20)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err.Error())
		return
	}
	if list == nil {
		list = []store.TrainRun{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": list})
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func TopClasses(probs map[string]float64, n int) []string {
	type kv struct {
		k string
		v float64
	}
	arr := make([]kv, 0, len(probs))
	for k, v := range probs {
		arr = append(arr, kv{k, v})
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].v > arr[j].v })
	if n > len(arr) {
		n = len(arr)
	}
	out := make([]string, n)
	for i := 0; i < n; i++ {
		out[i] = arr[i].k
	}
	return out
}
