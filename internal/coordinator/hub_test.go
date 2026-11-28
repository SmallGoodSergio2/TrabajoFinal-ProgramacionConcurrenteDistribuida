package coordinator

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/dataset"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/node"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
)

func syntheticCSV(t *testing.T) string {
	t.Helper()
	var sb strings.Builder
	sb.WriteString("ANIO,MES,DPTO_HECHO_NEW,PROV_HECHO,DIST_HECHO,UBIGEO_HECHO,P_MODALIDADES,cantidad\n")
	mods := []string{"Hurto", "Robo", "Estafa"}
	for d := 0; d < 150; d++ {
		for a := 2018; a <= 2021; a++ {
			for m := 1; m <= 12; m++ {
				for k, mod := range mods {
					fmt.Fprintf(&sb, "%d,%d,DPTO%d,PROV%d,DIST%d,%d,%s,%d\n",
						a, m, d%5, d%10, d, 10101+d, mod, 1+(d+m+k*7)%9)
				}
			}
		}
	}
	p := filepath.Join(t.TempDir(), "data.csv")
	if err := os.WriteFile(p, []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestStarTopologyFiveNodes(t *testing.T) {
	hub, err := Listen("127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer hub.Close()

	const nNodes = 5
	cancels := make([]context.CancelFunc, nNodes)
	for i := 0; i < nNodes; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancels[i] = cancel
		defer cancel()
		nd := &node.Node{ID: fmt.Sprintf("nodo-%d", i+1), Coord: hub.Addr().String(), Workers: 2}
		go nd.Run(ctx)
	}

	wctx, wcancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer wcancel()
	if err := hub.WaitForNodes(wctx, nNodes); err != nil {
		t.Fatal(err)
	}

	// Sin modelo, la predicción debe fallar de forma limpia (no colgarse).
	if _, err := hub.Predict(context.Background(), make([]float32, dataset.NumFeatures)); err == nil {
		t.Fatal("se esperaba error: los nodos aún no tienen modelo")
	}

	tctx, tcancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer tcancel()
	res, err := hub.Train(tctx, TrainConfig{
		Params:       rf.Params{NumTrees: 10, MaxDepth: 8, MinSamplesLeaf: 5},
		DataPath:     syntheticCSV(t),
		TestFraction: 0.1,
	})
	if err != nil {
		t.Fatal(err)
	}
	trees := 0
	for _, r := range res {
		trees += r.Trees
		if r.Rows == 0 {
			t.Fatalf("shard vacío en %s", r.NodeID)
		}
	}
	if trees != 10 {
		t.Fatalf("árboles entrenados = %d, se esperaban 10", trees)
	}

	x := []float32{2019, 6, 1, 101, 10110, 2.0}
	r, err := hub.Predict(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	if r.Nodes != nNodes || r.Trees != 10 {
		t.Fatalf("respondieron %d nodos / %d árboles", r.Nodes, r.Trees)
	}
	var s float64
	for _, p := range r.Probs {
		s += float64(p)
	}
	if math.Abs(s-1) > 1e-3 {
		t.Fatalf("las probabilidades suman %.4f, no 1", s)
	}

	// Tolerancia a fallos: se cae un nodo y el servicio sigue.
	cancels[0]()
	deadline := time.Now().Add(5 * time.Second)
	for hub.NodeCount() != nNodes-1 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if hub.NodeCount() != nNodes-1 {
		t.Fatalf("nodos activos = %d, se esperaban %d", hub.NodeCount(), nNodes-1)
	}
	r, err = hub.Predict(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	if r.Nodes != nNodes-1 || r.Trees >= 10 {
		t.Fatalf("tras la caída: %d nodos / %d árboles", r.Nodes, r.Trees)
	}
}
