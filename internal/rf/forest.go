package rf

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// Params son los hiperparámetros del bosque.
type Params struct {
	NumTrees       int     `json:"num_trees" bson:"num_trees"`
	MaxDepth       int     `json:"max_depth" bson:"max_depth"`
	MinSamplesLeaf int     `json:"min_samples_leaf" bson:"min_samples_leaf"`
	MaxFeatures    int     `json:"max_features" bson:"max_features"`
	MaxBins        int     `json:"max_bins" bson:"max_bins"`
	SampleRate     float64 `json:"sample_rate" bson:"sample_rate"`
	Seed           uint64  `json:"seed" bson:"seed"`
}

// WithDefaults completa los valores no definidos y los acota a rangos válidos.
func (p Params) WithDefaults(nFeatures int) Params {
	if p.NumTrees <= 0 {
		p.NumTrees = 30
	}
	if p.MaxDepth <= 0 {
		p.MaxDepth = 14
	}
	p.MaxDepth = min(p.MaxDepth, 30)
	if p.MinSamplesLeaf <= 0 {
		p.MinSamplesLeaf = 20
	}
	if p.MaxFeatures <= 0 {
		p.MaxFeatures = max(1, int(math.Ceil(math.Sqrt(float64(nFeatures)))))
	}
	p.MaxFeatures = min(p.MaxFeatures, nFeatures)
	if p.MaxBins <= 1 {
		p.MaxBins = 64
	}
	p.MaxBins = min(p.MaxBins, 255)
	if p.SampleRate <= 0 {
		p.SampleRate = 1
	}
	if p.Seed == 0 {
		p.Seed = 42
	}
	return p
}

// Forest es un conjunto de árboles; Importance acumula la reducción de
// impureza Gini por característica.
type Forest struct {
	Trees      []Tree
	NClasses   int
	NFeatures  int
	Importance []float64
}

// Train entrena p.NumTrees árboles en paralelo con un pool de workers.
// Cada árbol escribe en su propia posición del slice, por lo que no hay
// estado compartido mutable entre goroutines. progress se invoca desde
// varias goroutines y debe ser seguro para uso concurrente.
func Train(d *Dataset, p Params, workers int, progress func(done, total int)) *Forest {
	p = p.WithDefaults(d.NFeatures)
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	b := buildBins(d, p.MaxBins)

	trees := make([]Tree, p.NumTrees)
	imps := make([][]float64, p.NumTrees)
	jobs := make(chan int)
	var done atomic.Int32
	var wg sync.WaitGroup
	for w := 0; w < min(workers, p.NumTrees); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				trees[t], imps[t] = trainTree(d, b, p, p.Seed+uint64(t)*7919)
				n := done.Add(1)
				if progress != nil {
					progress(int(n), p.NumTrees)
				}
			}
		}()
	}
	for t := 0; t < p.NumTrees; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()

	f := &Forest{Trees: trees, NClasses: d.NClasses, NFeatures: d.NFeatures, Importance: make([]float64, d.NFeatures)}
	for _, imp := range imps {
		for i, v := range imp {
			f.Importance[i] += v
		}
	}
	return f
}

// Merged devuelve un bosque nuevo con los árboles de f y o, sin modificar
// ninguno de los dos (otras goroutines pueden estar prediciendo con f).
func (f *Forest) Merged(o *Forest) *Forest {
	m := &Forest{
		Trees:      make([]Tree, 0, len(f.Trees)+len(o.Trees)),
		NClasses:   f.NClasses,
		NFeatures:  f.NFeatures,
		Importance: make([]float64, len(f.Importance)),
	}
	m.Trees = append(append(m.Trees, f.Trees...), o.Trees...)
	for i := range m.Importance {
		m.Importance[i] = f.Importance[i] + o.Importance[i]
	}
	return m
}

// PredictBatch devuelve, para cada fila, la suma de las distribuciones de
// clase de todos los árboles (n*NClasses). El coordinador normaliza
// dividiendo entre el total de árboles de todos los nodos.
func (f *Forest) PredictBatch(x []float32, workers int) []float32 {
	nf, nc := f.NFeatures, f.NClasses
	n := len(x) / nf
	out := make([]float32, n*nc)
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	workers = max(1, min(workers, n/256+1))
	step := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo, hi := w*step, min((w+1)*step, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				row := x[i*nf : (i+1)*nf]
				dst := out[i*nc : (i+1)*nc]
				for t := range f.Trees {
					for k, v := range f.Trees[t].Predict(row) {
						dst[k] += v
					}
				}
			}
		}(lo, hi)
	}
	wg.Wait()
	return out
}
