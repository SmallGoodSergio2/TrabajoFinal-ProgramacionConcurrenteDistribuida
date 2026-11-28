package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/docs"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/api"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/coordinator"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/store"
)

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	httpAddr := flag.String("http", getenv("HTTP_ADDR", ":8080"), "dirección HTTP de la API")
	tcpAddr := flag.String("tcp", getenv("TCP_ADDR", ":9000"), "dirección TCP donde se registran los nodos")
	dataPath := flag.String("data", getenv("DATA_PATH", "Dataset/DenunciasPoliciales_1000000.csv"), "CSV del dataset")
	mongoURI := flag.String("mongo", getenv("MONGO_URI", "mongodb://localhost:27017"), "URI MongoDB (vacío = sin persistencia)")
	redisAddr := flag.String("redis", getenv("REDIS_ADDR", "localhost:6379"), "addr Redis host:port (vacío = sin caché)")
	workers := flag.Int("workers", runtime.NumCPU(), "goroutines para cargar el dataset")
	waitNodes := flag.Int("nodes", 0, "si >0, espera N nodos al arrancar")
	trees := flag.Int("trees", 30, "árboles totales al auto-entrenar")
	autoTrain := flag.Bool("auto-train", false, "entrenar automáticamente tras conectar nodos")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	hub, err := coordinator.Listen(*tcpAddr)
	if err != nil {
		log.Fatal(err)
	}
	defer hub.Close()
	log.Printf("[api] hub TCP en %s", hub.Addr())

	stCtx, stCancel := context.WithTimeout(ctx, 15*time.Second)
	st, err := store.Open(stCtx, *mongoURI, *redisAddr)
	stCancel()
	if err != nil {
		log.Printf("[api] aviso store: %v (continúa sin Mongo/Redis)", err)
		st, _ = store.Open(context.Background(), "", "")
	}
	defer st.Close(context.Background())

	srvAPI := api.New(hub, st, api.Config{
		HTTPAddr:     *httpAddr,
		DataPath:     *dataPath,
		Workers:      *workers,
		TestFraction: 0.1,
		DefaultTrees: *trees,
		SwaggerUI:    docs.SwaggerHTML,
		OpenAPI:      docs.OpenAPI,
	})

	go func() {
		if err := srvAPI.LoadData(*workers); err != nil {
			log.Printf("[api] carga dataset: %v", err)
		}
	}()

	if *waitNodes > 0 {
		go func() {
			wctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			defer cancel()
			log.Printf("[api] esperando %d nodos...", *waitNodes)
			if err := hub.WaitForNodes(wctx, *waitNodes); err != nil {
				log.Printf("[api] espera nodos: %v", err)
				return
			}
			if !*autoTrain {
				return
			}
			log.Printf("[api] auto-entrenamiento (%d árboles)...", *trees)
			tctx, tcancel := context.WithTimeout(ctx, 45*time.Minute)
			defer tcancel()
			begin := time.Now()
			res, err := hub.Train(tctx, coordinator.TrainConfig{
				Params:       rf.Params{NumTrees: *trees},
				DataPath:     *dataPath,
				TestFraction: 0.1,
			})
			run := &store.TrainRun{
				TreesTotal: *trees, Nodes: len(res),
				DurationMs: float64(time.Since(begin).Microseconds()) / 1000,
				DataPath:   *dataPath, CreatedAt: time.Now().UTC(),
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
			_ = st.SaveTrainRun(context.Background(), run)
			if err != nil {
				log.Printf("[api] auto-train incompleto: %v (%.1fs)", err, time.Since(begin).Seconds())
			} else {
				log.Printf("[api] auto-train OK en %.1fs", time.Since(begin).Seconds())
			}
		}()
	}

	httpSrv := &http.Server{Addr: *httpAddr, Handler: srvAPI.Handler()}
	go func() {
		log.Printf("[api] HTTP en %s (Swagger: /swagger/)", *httpAddr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = httpSrv.Shutdown(shutdown)
}
