package rf

import (
	"math/rand/v2"
	"sync/atomic"
	"testing"
)

// La clase depende de dos características con ruido; un bosque debe
// superar claramente al azar (1/3).
func syntheticDataset(n int, seed uint64) *Dataset {
	r := rand.New(rand.NewPCG(seed, 1))
	d := &Dataset{NFeatures: 3, NClasses: 3}
	for i := 0; i < n; i++ {
		a, b, noise := r.Float32()*10, r.Float32()*10, r.Float32()
		class := uint8(0)
		switch {
		case a > 6:
			class = 1
		case b > 5:
			class = 2
		}
		if r.Float32() < 0.05 {
			class = uint8(r.IntN(3))
		}
		d.X = append(d.X, a, b, noise)
		d.Y = append(d.Y, class)
		d.W = append(d.W, float32(1+r.IntN(3)))
	}
	return d
}

func accuracy(f *Forest, d *Dataset) float64 {
	probs := f.PredictBatch(d.X, 4)
	correct := 0
	for i := range d.Y {
		best := 0
		for k := 1; k < d.NClasses; k++ {
			if probs[i*d.NClasses+k] > probs[i*d.NClasses+best] {
				best = k
			}
		}
		if uint8(best) == d.Y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(d.Y))
}

func TestForestLearns(t *testing.T) {
	train, test := syntheticDataset(20000, 1), syntheticDataset(5000, 2)
	var calls atomic.Int32
	f := Train(train, Params{NumTrees: 12, MaxDepth: 8}, 4, func(done, total int) { calls.Add(1) })
	if len(f.Trees) != 12 || calls.Load() != 12 {
		t.Fatalf("árboles=%d progreso=%d", len(f.Trees), calls.Load())
	}
	if acc := accuracy(f, test); acc < 0.9 {
		t.Fatalf("accuracy %.3f demasiado baja", acc)
	}
	if f.Importance[2] > f.Importance[0] || f.Importance[2] > f.Importance[1] {
		t.Fatalf("la característica de ruido no debería ser la más importante: %v", f.Importance)
	}
}

// Dos bosques entrenados en shards distintos y fusionados deben votar igual
// de bien que uno entrenado con todos los datos.
func TestMergedShards(t *testing.T) {
	full := syntheticDataset(20000, 3)
	half := func(lo, hi int) *Dataset {
		return &Dataset{X: full.X[lo*3 : hi*3], Y: full.Y[lo:hi], W: full.W[lo:hi], NFeatures: 3, NClasses: 3}
	}
	a := Train(half(0, 10000), Params{NumTrees: 6, Seed: 1}, 2, nil)
	b := Train(half(10000, 20000), Params{NumTrees: 6, Seed: 2}, 2, nil)
	m := a.Merged(b)
	if len(m.Trees) != 12 || len(a.Trees) != 6 {
		t.Fatalf("Merged no debe modificar el original: merged=%d a=%d", len(m.Trees), len(a.Trees))
	}
	if acc := accuracy(m, syntheticDataset(5000, 4)); acc < 0.9 {
		t.Fatalf("accuracy %.3f demasiado baja", acc)
	}
}
