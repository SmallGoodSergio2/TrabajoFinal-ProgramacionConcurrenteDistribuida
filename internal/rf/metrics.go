package rf

import (
	"math"
	"slices"
)

// Metrics se calculan ponderando cada fila por su peso (número de denuncias).
type Metrics struct {
	Accuracy     float64
	Top3Accuracy float64
	LogLoss      float64
	// Confusion[real][predicha] acumula pesos.
	Confusion [][]float64
}

// Evaluate predice el conjunto d con el bosque y calcula las métricas.
func Evaluate(f *Forest, d *Dataset, workers int) Metrics {
	nc := d.NClasses
	sums := f.PredictBatch(d.X, workers)
	m := Metrics{Confusion: make([][]float64, nc)}
	for k := range m.Confusion {
		m.Confusion[k] = make([]float64, nc)
	}
	inv := 1 / float64(len(f.Trees))
	order := make([]int, nc)
	var total, correct, top3, logLoss float64
	for i, truth := range d.Y {
		w := float64(d.W[i])
		p := sums[i*nc : (i+1)*nc]
		for k := range order {
			order[k] = k
		}
		slices.SortFunc(order, func(a, b int) int {
			switch {
			case p[a] > p[b]:
				return -1
			case p[a] < p[b]:
				return 1
			}
			return a - b
		})
		total += w
		if order[0] == int(truth) {
			correct += w
		}
		if slices.Contains(order[:min(3, nc)], int(truth)) {
			top3 += w
		}
		logLoss -= w * math.Log(math.Max(float64(p[truth])*inv, 1e-7))
		m.Confusion[truth][order[0]] += w
	}
	if total > 0 {
		m.Accuracy, m.Top3Accuracy, m.LogLoss = correct/total, top3/total, logLoss/total
	}
	return m
}

// Baseline evalúa el modelo trivial que siempre predice la clase más
// frecuente del entrenamiento y usa la distribución global como probabilidad.
func Baseline(train, test *Dataset) (class int, accuracy, logLoss float64) {
	prior := make([]float64, train.NClasses)
	total := 0.0
	for i, y := range train.Y {
		prior[y] += float64(train.W[i])
		total += float64(train.W[i])
	}
	for k := range prior {
		prior[k] /= total
		if prior[k] > prior[class] {
			class = k
		}
	}
	var w, correct float64
	for i, y := range test.Y {
		wi := float64(test.W[i])
		w += wi
		if int(y) == class {
			correct += wi
		}
		logLoss -= wi * math.Log(math.Max(prior[y], 1e-7))
	}
	return class, correct / w, logLoss / w
}
