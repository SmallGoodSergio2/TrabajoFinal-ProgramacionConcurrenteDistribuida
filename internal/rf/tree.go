// Package rf implementa un Random Forest de clasificación con árboles CART
// ponderados. La búsqueda de cortes usa histogramas sobre características
// discretizadas, lo que mantiene el costo lineal en el número de filas.
package rf

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
)

// Dataset almacena las filas en formato fila-mayor: X[i*NFeatures+f].
type Dataset struct {
	X         []float32
	Y         []uint8
	W         []float32
	NFeatures int
	NClasses  int
}

func (d *Dataset) Rows() int { return len(d.Y) }

// Node es un nodo del árbol; Feature < 0 indica hoja.
type Node struct {
	Feature   int
	Threshold float32
	Left      int32
	Right     int32
	Value     []float32
}

type Tree struct {
	Nodes []Node
}

// Predict devuelve la distribución de clases de la hoja alcanzada.
func (t *Tree) Predict(x []float32) []float32 {
	i := 0
	for {
		n := &t.Nodes[i]
		if n.Feature < 0 {
			return n.Value
		}
		if x[n.Feature] <= n.Threshold {
			i = int(n.Left)
		} else {
			i = int(n.Right)
		}
	}
}

// binned guarda cada característica discretizada en a lo sumo MaxBins
// intervalos. edges[f][b] es el valor máximo del intervalo b.
type binned struct {
	bins  [][]uint8
	edges [][]float32
}

func buildBins(d *Dataset, maxBins int) *binned {
	n, nf := d.Rows(), d.NFeatures
	b := &binned{bins: make([][]uint8, nf), edges: make([][]float32, nf)}
	done := make(chan struct{}, nf)
	for f := 0; f < nf; f++ {
		go func(f int) {
			defer func() { done <- struct{}{} }()
			col := make([]float32, n)
			for i := 0; i < n; i++ {
				col[i] = d.X[i*nf+f]
			}
			sorted := slices.Clone(col)
			slices.Sort(sorted)
			edges := quantileEdges(sorted, maxBins)
			bins := make([]uint8, n)
			for i, v := range col {
				bins[i] = uint8(sort.Search(len(edges), func(k int) bool { return edges[k] >= v }))
			}
			b.bins[f], b.edges[f] = bins, edges
		}(f)
	}
	for f := 0; f < nf; f++ {
		<-done
	}
	return b
}

func quantileEdges(sorted []float32, maxBins int) []float32 {
	var distinct []float32
	for i, v := range sorted {
		if i == 0 || v != sorted[i-1] {
			distinct = append(distinct, v)
		}
	}
	if len(distinct) <= maxBins {
		return distinct
	}
	edges := make([]float32, 0, maxBins)
	n := len(sorted)
	for k := 1; k < maxBins; k++ {
		v := sorted[k*n/maxBins-1]
		if len(edges) == 0 || v > edges[len(edges)-1] {
			edges = append(edges, v)
		}
	}
	if last := sorted[n-1]; edges[len(edges)-1] < last {
		edges = append(edges, last)
	}
	return edges
}

type builder struct {
	d          *Dataset
	b          *binned
	p          Params
	rng        *rand.Rand
	w          []float64
	hist       []float64
	cnt        []int
	left       []float64
	right      []float64
	nodes      []Node
	importance []float64
}

// trainTree entrena un árbol sobre una muestra bootstrap de Poisson
// ponderada por el número de denuncias de cada fila.
func trainTree(d *Dataset, b *binned, p Params, seed uint64) (Tree, []float64) {
	n, nc := d.Rows(), d.NClasses
	bld := &builder{
		d:          d,
		b:          b,
		p:          p,
		rng:        rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)),
		w:          make([]float64, n),
		hist:       make([]float64, 256*nc),
		cnt:        make([]int, 256),
		left:       make([]float64, nc),
		right:      make([]float64, nc),
		importance: make([]float64, d.NFeatures),
	}
	idx := make([]int32, 0, n)
	for i := 0; i < n; i++ {
		if d.W[i] <= 0 {
			continue
		}
		k := poisson(bld.rng, p.SampleRate)
		if k == 0 {
			continue
		}
		bld.w[i] = float64(k) * float64(d.W[i])
		idx = append(idx, int32(i))
	}
	bld.grow(idx, 0)
	return Tree{Nodes: bld.nodes}, bld.importance
}

func poisson(r *rand.Rand, lambda float64) int {
	limit := math.Exp(-lambda)
	k, prod := 0, r.Float64()
	for prod > limit {
		k++
		prod *= r.Float64()
	}
	return k
}

func sumSquares(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	return s
}

func (bld *builder) leaf(id int, totals []float64, wsum float64) {
	val := make([]float32, len(totals))
	if wsum > 0 {
		for k, t := range totals {
			val[k] = float32(t / wsum)
		}
	}
	bld.nodes[id].Value = val
}

func (bld *builder) grow(idx []int32, depth int) int32 {
	id := len(bld.nodes)
	bld.nodes = append(bld.nodes, Node{Feature: -1})
	nc := bld.d.NClasses
	y := bld.d.Y

	totals := make([]float64, nc)
	wsum := 0.0
	for _, i := range idx {
		totals[y[i]] += bld.w[i]
		wsum += bld.w[i]
	}
	nonZero := 0
	for _, t := range totals {
		if t > 0 {
			nonZero++
		}
	}
	minLeaf := bld.p.MinSamplesLeaf
	if depth >= bld.p.MaxDepth || len(idx) < 2*minLeaf || nonZero <= 1 {
		bld.leaf(id, totals, wsum)
		return int32(id)
	}

	parentScore := sumSquares(totals) / wsum
	bestGain, bestF, bestBin := 1e-9*wsum, -1, -1
	features := bld.rng.Perm(bld.d.NFeatures)[:bld.p.MaxFeatures]
	for _, f := range features {
		edges := bld.b.edges[f]
		nb := len(edges)
		if nb < 2 {
			continue
		}
		hist, cnt := bld.hist[:nb*nc], bld.cnt[:nb]
		clear(hist)
		clear(cnt)
		col := bld.b.bins[f]
		for _, i := range idx {
			bi := int(col[i])
			hist[bi*nc+int(y[i])] += bld.w[i]
			cnt[bi]++
		}
		clear(bld.left)
		wl, cl := 0.0, 0
		for bi := 0; bi < nb-1; bi++ {
			for k := 0; k < nc; k++ {
				h := hist[bi*nc+k]
				bld.left[k] += h
				wl += h
			}
			cl += cnt[bi]
			if cl < minLeaf {
				continue
			}
			if len(idx)-cl < minLeaf {
				break
			}
			wr := wsum - wl
			if wl <= 0 || wr <= 0 {
				continue
			}
			for k := 0; k < nc; k++ {
				bld.right[k] = totals[k] - bld.left[k]
			}
			gain := sumSquares(bld.left)/wl + sumSquares(bld.right)/wr - parentScore
			if gain > bestGain {
				bestGain, bestF, bestBin = gain, f, bi
			}
		}
	}
	if bestF < 0 {
		bld.leaf(id, totals, wsum)
		return int32(id)
	}

	col := bld.b.bins[bestF]
	lo, hi := 0, len(idx)-1
	for lo <= hi {
		if int(col[idx[lo]]) <= bestBin {
			lo++
		} else {
			idx[lo], idx[hi] = idx[hi], idx[lo]
			hi--
		}
	}
	bld.importance[bestF] += bestGain
	bld.nodes[id].Feature = bestF
	bld.nodes[id].Threshold = bld.b.edges[bestF][bestBin]
	left := bld.grow(idx[:lo], depth+1)
	right := bld.grow(idx[lo:], depth+1)
	bld.nodes[id].Left = left
	bld.nodes[id].Right = right
	return int32(id)
}
