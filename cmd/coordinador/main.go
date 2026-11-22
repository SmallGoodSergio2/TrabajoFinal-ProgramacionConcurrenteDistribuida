package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/coordinator"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/dataset"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
)

func main() {
	listen := flag.String("listen", ":9000", "dirección TCP donde se registran los nodos")
	nodes := flag.Int("nodes", 5, "nodos a esperar antes de entrenar")
	wait := flag.Duration("wait", 2*time.Minute, "tiempo máximo esperando a los nodos")
	path := flag.String("data", "Dataset/DenunciasPoliciales_1000000.csv", "CSV (la misma ruta debe ser válida en cada nodo)")
	trees := flag.Int("trees", 30, "árboles totales del bosque (se reparten entre nodos)")
	depth := flag.Int("depth", 14, "profundidad máxima")
	leaf := flag.Int("leaf", 20, "mínimo de filas por hoja")
	testFrac := flag.Float64("test", 0.1, "fracción de grupos distrito-mes para prueba")
	benchN := flag.Int("bench", 500, "predicciones individuales para medir latencia")
	conc := flag.Int("conc", 8, "clientes simultáneos en la prueba de latencia")
	workers := flag.Int("workers", runtime.NumCPU(), "goroutines para cargar el dataset local")
	hold := flag.Bool("hold", false, "seguir ejecutando (mostrando el estado de los nodos) al terminar")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	hub, err := coordinator.Listen(*listen)
	if err != nil {
		log.Fatal(err)
	}
	defer hub.Close()
	log.Printf("coordinador en %s; esperando %d nodos...", hub.Addr(), *nodes)

	wctx, cancel := context.WithTimeout(ctx, *wait)
	err = hub.WaitForNodes(wctx, *nodes)
	cancel()
	if err != nil {
		log.Fatal(err)
	}
	printNodes(hub)

	// El coordinador también carga el dataset (en paralelo con el entrenamiento
	// de los nodos) porque necesita armar los vectores de características.
	type loaded struct {
		data *dataset.Prepared
		err  error
	}
	loadCh := make(chan loaded, 1)
	go func() {
		recs, st, err := dataset.LoadCSV(*path, *workers)
		if err != nil {
			loadCh <- loaded{err: err}
			return
		}
		loadCh <- loaded{data: dataset.Prepare(recs, st, *testFrac, *workers)}
	}()

	fmt.Println("\n== Entrenamiento distribuido ==")
	begin := time.Now()
	tctx, tcancel := context.WithTimeout(ctx, 30*time.Minute)
	res, err := hub.Train(tctx, coordinator.TrainConfig{
		Params:       rf.Params{NumTrees: *trees, MaxDepth: *depth, MinSamplesLeaf: *leaf},
		DataPath:     *path,
		TestFraction: *testFrac,
	})
	tcancel()
	fmt.Printf("%-16s %6s %10s %7s %10s\n", "nodo", "shard", "filas", "árboles", "ms")
	for _, r := range res {
		if r.Err != nil {
			fmt.Printf("%-16s %6d  ERROR: %v\n", r.NodeID, r.Shard, r.Err)
			continue
		}
		fmt.Printf("%-16s %6d %10d %7d %10.0f\n", r.NodeID, r.Shard, r.Rows, r.Trees, r.Millis)
	}
	fmt.Printf("tiempo total (pared) = %.1f s con %d nodos\n", time.Since(begin).Seconds(), len(res))
	if err != nil {
		log.Fatalf("entrenamiento incompleto: %v", err)
	}

	l := <-loadCh
	if l.err != nil {
		log.Fatal(l.err)
	}
	data := l.data
	x, y, w := data.Matrix(data.Test)
	if len(y) > 0 {
		evaluate(ctx, hub, x, y, w, len(data.Classes))
		latency(hub, x, *benchN, *conc)
	}
	printNodes(hub)

	if *hold {
		log.Println("coordinador en espera (Ctrl+C para salir)")
		t := time.NewTicker(10 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				printNodes(hub)
			}
		}
	}
}

func printNodes(hub *coordinator.Hub) {
	fmt.Println("\nNodos conectados:")
	for _, n := range hub.Status() {
		fmt.Printf("  %-16s %-22s núcleos=%d árboles=%d rtt=%.2f ms visto hace %d ms\n",
			n.ID, n.Addr, n.Cores, n.Trees, n.RTTMs, n.LastSeenMs)
	}
}

// evaluate predice todo el conjunto de prueba a través del clúster, en lotes,
// y calcula la exactitud ponderada (igual que cmd/entrenar).
func evaluate(ctx context.Context, hub *coordinator.Hub, x []float32, y []uint8, w []float32, nc int) {
	const batch = 2000
	nf := dataset.NumFeatures
	var hit, tot float64
	begin := time.Now()
	for lo := 0; lo < len(y); lo += batch {
		hi := min(lo+batch, len(y))
		cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		r, err := hub.Predict(cctx, x[lo*nf:hi*nf])
		cancel()
		if err != nil {
			log.Fatalf("evaluación: %v", err)
		}
		for i := lo; i < hi; i++ {
			p := r.Probs[(i-lo)*nc : (i-lo+1)*nc]
			best := 0
			for k := range p {
				if p[k] > p[best] {
					best = k
				}
			}
			tot += float64(w[i])
			if best == int(y[i]) {
				hit += float64(w[i])
			}
		}
	}
	fmt.Println("\n== Evaluación distribuida ==")
	if tot > 0 {
		fmt.Printf("filas de prueba=%d accuracy=%.2f%% (%.1f s)\n", len(y), 100*hit/tot, time.Since(begin).Seconds())
	}
}

// latency mide predicciones de UNA fila (lo que hará el API) con `conc`
// clientes concurrentes y reporta percentiles.
func latency(hub *coordinator.Hub, x []float32, total, conc int) {
	nf := dataset.NumFeatures
	rows := len(x) / nf
	if rows == 0 || total <= 0 {
		return
	}
	one := func(i int) (time.Duration, error) {
		j := rand.IntN(rows)
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		t := time.Now()
		_, err := hub.Predict(ctx, x[j*nf:(j+1)*nf])
		return time.Since(t), err
	}
	for i := 0; i < 20; i++ { // calentamiento
		one(i)
	}

	lat := make([]time.Duration, total)
	var errs atomic.Int64
	jobs := make(chan int)
	var wg sync.WaitGroup
	begin := time.Now()
	for c := 0; c < max(1, conc); c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				d, err := one(i)
				lat[i] = d
				if err != nil {
					errs.Add(1)
				}
			}
		}()
	}
	for i := 0; i < total; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	elapsed := time.Since(begin)

	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	pct := func(p float64) float64 { return float64(lat[int(float64(len(lat)-1)*p)].Microseconds()) / 1000 }
	fmt.Printf("\n== Latencia de predicción (fila única, %d peticiones, %d clientes) ==\n", total, conc)
	fmt.Printf("p50=%.2f ms  p95=%.2f ms  p99=%.2f ms  max=%.2f ms  errores=%d  throughput=%.0f req/s\n",
		pct(0.50), pct(0.95), pct(0.99), pct(1), errs.Load(), float64(total)/elapsed.Seconds())
	if pct(0.99) < 100 {
		fmt.Println("OK: p99 < 100 ms (requisito del enunciado)")
	} else {
		fmt.Println("AVISO: p99 >= 100 ms")
	}
}
