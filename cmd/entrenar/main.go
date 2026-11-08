// Comando entrenar: carga y limpia el dataset de forma concurrente, muestra
// el análisis descriptivo, entrena el Random Forest en paralelo y lo evalúa.
package main

import (
	"cmp"
	"flag"
	"fmt"
	"os"
	"runtime"
	"slices"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/dataset"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
)

var meses = []string{"Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"}

func main() {
	path := flag.String("data", "Dataset/DenunciasPoliciales_1000000.csv", "ruta del CSV")
	workers := flag.Int("workers", runtime.NumCPU(), "goroutines para carga y entrenamiento")
	trees := flag.Int("trees", 30, "número de árboles")
	depth := flag.Int("depth", 14, "profundidad máxima")
	leaf := flag.Int("leaf", 20, "mínimo de filas por hoja")
	testFraction := flag.Float64("test", 0.1, "fracción de grupos distrito-mes para prueba")
	flag.Parse()

	fmt.Println("== Parte A: carga concurrente y limpieza ==")
	records, stats, err := dataset.LoadCSV(*path, *workers)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Printf("líneas=%d válidas=%d inválidas=%d duplicadas=%d\n", stats.Lines, stats.Valid, stats.Invalid, stats.Duplicates)
	fmt.Printf("workers=%d bloques=%d tiempo=%.1f ms (%.0f filas/s)\n", stats.Workers, stats.Chunks, stats.DurationMs, stats.RowsPerSec)

	data := dataset.Prepare(records, stats, *testFraction, *workers)
	agg := data.Aggregates
	fmt.Println("\n== Análisis descriptivo ==")
	fmt.Printf("denuncias=%d registros=%d distritos=%d años=%d-%d clases=%d\n",
		agg.TotalDenuncias, agg.Registros, len(data.Districts), data.MinYear, data.MaxYear, len(data.Classes))
	mods := slices.Clone(data.Classes)
	slices.SortFunc(mods, func(a, b string) int { return cmp.Compare(agg.PorModalidad[b], agg.PorModalidad[a]) })
	for _, m := range mods {
		v := agg.PorModalidad[m]
		fmt.Printf("  %-42s %9d  %5.1f%%\n", m, v, 100*float64(v)/float64(agg.TotalDenuncias))
	}
	fmt.Println("distritos con más denuncias:")
	for _, d := range agg.TopDistritos[:5] {
		fmt.Printf("  %-25s %-25s %d\n", d.Dist, d.Dpto, d.Total)
	}
	fmt.Print("estacionalidad (años completos):")
	for i, v := range data.Seasonality {
		fmt.Printf(" %s=%.3f", meses[i], v)
	}
	fmt.Println()

	toDataset := func(idx []int) *rf.Dataset {
		x, y, w := data.Matrix(idx)
		return &rf.Dataset{X: x, Y: y, W: w, NFeatures: dataset.NumFeatures, NClasses: len(data.Classes)}
	}
	train, test := toDataset(data.Train), toDataset(data.Test)
	params := rf.Params{NumTrees: *trees, MaxDepth: *depth, MinSamplesLeaf: *leaf}.WithDefaults(dataset.NumFeatures)

	fmt.Println("\n== Parte B: entrenamiento paralelo del Random Forest ==")
	fmt.Printf("entrenamiento=%d filas prueba=%d filas parámetros=%+v\n", train.Rows(), test.Rows(), params)
	begin := time.Now()
	forest := rf.Train(train, params, *workers, func(done, total int) {
		fmt.Printf("\r  árboles %d/%d", done, total)
	})
	fmt.Printf("\ntiempo de entrenamiento=%.1f ms con %d goroutines\n", float64(time.Since(begin).Microseconds())/1000, *workers)

	fmt.Println("\n== Evaluación (ponderada por número de denuncias) ==")
	m := rf.Evaluate(forest, test, *workers)
	base, baseAcc, baseLoss := rf.Baseline(train, test)
	fmt.Printf("accuracy=%.2f%%  top3=%.2f%%  logloss=%.3f\n", 100*m.Accuracy, 100*m.Top3Accuracy, m.LogLoss)
	fmt.Printf("línea base (%s): accuracy=%.2f%%  logloss=%.3f\n", data.Classes[base], 100*baseAcc, baseLoss)
	fmt.Println("por clase (precisión / recall):")
	for k, name := range data.Classes {
		var support, predicted float64
		for j := range data.Classes {
			support += m.Confusion[k][j]
			predicted += m.Confusion[j][k]
		}
		prec, rec := 0.0, 0.0
		if predicted > 0 {
			prec = m.Confusion[k][k] / predicted
		}
		if support > 0 {
			rec = m.Confusion[k][k] / support
		}
		fmt.Printf("  %-42s %5.1f%% / %5.1f%%\n", name, 100*prec, 100*rec)
	}
	fmt.Println("matriz de confusión (% por fila real; columnas en el mismo orden):")
	for k, name := range data.Classes {
		var support float64
		for _, v := range m.Confusion[k] {
			support += v
		}
		fmt.Printf("  %-42s", name)
		for _, v := range m.Confusion[k] {
			fmt.Printf(" %5.1f", 100*v/support)
		}
		fmt.Println()
	}
	fmt.Println("importancia de variables (Gini):")
	total := 0.0
	for _, v := range forest.Importance {
		total += v
	}
	order := make([]int, len(forest.Importance))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int { return cmp.Compare(forest.Importance[b], forest.Importance[a]) })
	for _, f := range order {
		fmt.Printf("  %-18s %5.1f%%\n", dataset.FeatureNames[f], 100*forest.Importance[f]/total)
	}
}
