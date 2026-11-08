// Comando bench: mide el speedup del cargador concurrente y del
// entrenamiento paralelo frente a su ejecución con un solo worker.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/dataset"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
)

func median(ds []time.Duration) time.Duration {
	s := slices.Clone(ds)
	slices.Sort(s)
	return s[len(s)/2]
}

func measure(reps int, fn func()) time.Duration {
	ds := make([]time.Duration, reps)
	for i := range ds {
		begin := time.Now()
		fn()
		ds[i] = time.Since(begin)
	}
	return median(ds)
}

func row(label string, d, base time.Duration) {
	fmt.Printf("| %-22s | %10.1f | %7.2fx |\n", label, float64(d.Microseconds())/1000, float64(base)/float64(d))
}

func header(title, col string) {
	fmt.Printf("\n### %s\n\n| %-22s | %10s | %8s |\n|%s|%s|%s|\n", title, col, "ms (mediana)", "speedup", "---", "---:", "---:")
}

func main() {
	path := flag.String("data", "Dataset/DenunciasPoliciales_1000000.csv", "ruta del CSV")
	reps := flag.Int("reps", 3, "repeticiones por medición")
	trees := flag.Int("trees", 24, "árboles por entrenamiento")
	flag.Parse()

	fmt.Printf("CPU lógicas: %d · GOMAXPROCS: %d · Go %s\n", runtime.NumCPU(), runtime.GOMAXPROCS(0), runtime.Version())

	header("Parte A: carga concurrente de 1M de registros", "workers")
	var records []dataset.Record
	var stats dataset.LoadStats
	var base time.Duration
	for _, w := range []int{1, 2, 4, 8} {
		d := measure(*reps, func() {
			var err error
			records, stats, err = dataset.LoadCSV(*path, w)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		})
		if w == 1 {
			base = d
		}
		row(fmt.Sprintf("%d", w), d, base)
	}
	fmt.Printf("\nLíneas: %d · válidas: %d · inválidas: %d · duplicadas: %d\n", stats.Lines, stats.Valid, stats.Invalid, stats.Duplicates)

	data := dataset.Prepare(records, stats, 0.1, 0)
	idx := slices.Clone(data.Train)
	rng := rand.New(rand.NewPCG(42, 7))
	rng.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	x, y, w := data.Matrix(idx)
	full := &rf.Dataset{X: x, Y: y, W: w, NFeatures: dataset.NumFeatures, NClasses: len(data.Classes)}
	params := rf.Params{NumTrees: *trees, MaxDepth: 14, MinSamplesLeaf: 20}

	header(fmt.Sprintf("Parte B: entrenamiento paralelo de %d árboles (%d filas)", *trees, full.Rows()), "goroutines")
	for _, g := range []int{1, 2, 4, 8} {
		d := measure(*reps, func() { rf.Train(full, params, g, nil) })
		if g == 1 {
			base = d
		}
		row(fmt.Sprintf("%d", g), d, base)
	}
}
