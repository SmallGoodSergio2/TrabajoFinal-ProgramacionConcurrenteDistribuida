package node

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"log"
	"net"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/dataset"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/proto"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
)

type Node struct {
	ID      string
	Coord   string // host:puerto del coordinador
	Workers int    // goroutines para cargar y entrenar (0 = NumCPU)

	forest   atomic.Pointer[rf.Forest] // se reemplaza entero al terminar de entrenar
	training atomic.Bool

	cacheMu  sync.Mutex
	cacheKey string
	cacheDS  *rf.Dataset
}

// Run mantiene la conexión con el coordinador, reintentando con backoff si
// se cae o aún no está disponible. Termina cuando ctx se cancela.
func (n *Node) Run(ctx context.Context) error {
	backoff := time.Second
	for ctx.Err() == nil {
		registered, err := n.session(ctx)
		if ctx.Err() != nil {
			break
		}
		if registered {
			backoff = time.Second
		}
		log.Printf("[%s] sesión terminada: %v; reintentando en %s", n.ID, err, backoff)
		select {
		case <-ctx.Done():
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, 5*time.Second)
	}
	return ctx.Err()
}

func (n *Node) session(ctx context.Context) (registered bool, err error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", n.Coord)
	if err != nil {
		return false, err
	}
	defer conn.Close()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()

	enc := gob.NewEncoder(conn)
	dec := gob.NewDecoder(conn)
	var wmu sync.Mutex // varias goroutines responden por el mismo socket
	send := func(m proto.Msg) {
		wmu.Lock()
		defer wmu.Unlock()
		if err := enc.Encode(m); err != nil {
			conn.Close()
		}
	}

	send(proto.Msg{Kind: proto.Register, NodeID: n.ID, Cores: runtime.NumCPU()})
	log.Printf("[%s] registrado en %s", n.ID, n.Coord)

	for {
		var m proto.Msg // nuevo en cada vuelta (gob no limpia campos omitidos)
		if err := dec.Decode(&m); err != nil {
			return true, err
		}
		switch m.Kind {
		case proto.Ping:
			send(proto.Msg{Kind: proto.Pong})
		case proto.Train:
			// En goroutine aparte para seguir contestando Ping mientras entrena.
			go func() { send(n.handleTrain(m)) }()
		case proto.Predict:
			go func() { send(n.handlePredict(m)) }()
		}
	}
}

func errMsg(reqID uint64, err error) proto.Msg {
	return proto.Msg{Kind: proto.Err, ReqID: reqID, Err: err.Error()}
}

func (n *Node) handleTrain(m proto.Msg) (out proto.Msg) {
	defer func() {
		if r := recover(); r != nil {
			out = errMsg(m.ReqID, fmt.Errorf("panic: %v", r))
		}
	}()
	if !n.training.CompareAndSwap(false, true) {
		return errMsg(m.ReqID, errors.New("ya hay un entrenamiento en curso"))
	}
	defer n.training.Store(false)

	begin := time.Now()
	ds, err := n.shardDataset(m)
	if err != nil {
		return errMsg(m.ReqID, err)
	}
	log.Printf("[%s] shard %d/%d: %d filas, entrenando %d árboles", n.ID, m.Shard, m.NShards, ds.Rows(), m.Params.NumTrees)
	f := rf.Train(ds, m.Params, n.Workers, nil)
	n.forest.Store(f) // swap atómico: las predicciones en curso siguen con el bosque anterior
	ms := float64(time.Since(begin).Microseconds()) / 1000
	log.Printf("[%s] listo: %d árboles en %.0f ms", n.ID, len(f.Trees), ms)
	return proto.Msg{Kind: proto.TrainDone, ReqID: m.ReqID, NTrees: len(f.Trees), Rows: ds.Rows(), Millis: ms}
}

// shardDataset carga el CSV (concurrentemente), lo prepara igual que el resto
// de nodos y se queda solo con las filas de entrenamiento de su shard.
// El resultado se cachea para reentrenar con otros hiperparámetros sin recargar.
func (n *Node) shardDataset(m proto.Msg) (*rf.Dataset, error) {
	key := fmt.Sprintf("%s|%g|%d/%d", m.DataPath, m.TestFraction, m.Shard, m.NShards)
	n.cacheMu.Lock()
	defer n.cacheMu.Unlock()
	if n.cacheKey == key && n.cacheDS != nil {
		return n.cacheDS, nil
	}

	records, stats, err := dataset.LoadCSV(m.DataPath, n.Workers)
	if err != nil {
		return nil, fmt.Errorf("cargar %s: %w", m.DataPath, err)
	}
	data := dataset.Prepare(records, stats, m.TestFraction, n.Workers)

	idx := make([]int, 0, len(data.Train)/max(1, m.NShards)+1)
	for _, i := range data.Train {
		r := &data.Records[i]
		if dataset.ShardOf(dataset.MonthKey{Ubigeo: r.Ubigeo, Anio: r.Anio, Mes: r.Mes}, m.NShards) == m.Shard {
			idx = append(idx, i)
		}
	}
	if len(idx) == 0 {
		return nil, fmt.Errorf("el shard %d/%d quedó vacío", m.Shard, m.NShards)
	}
	x, y, w := data.Matrix(idx)
	ds := &rf.Dataset{X: x, Y: y, W: w, NFeatures: dataset.NumFeatures, NClasses: len(data.Classes)}
	n.cacheKey, n.cacheDS = key, ds
	return ds, nil
}

func (n *Node) handlePredict(m proto.Msg) (out proto.Msg) {
	defer func() {
		if r := recover(); r != nil {
			out = errMsg(m.ReqID, fmt.Errorf("panic: %v", r))
		}
	}()
	f := n.forest.Load()
	if f == nil {
		return errMsg(m.ReqID, errors.New("el nodo aún no tiene un modelo entrenado"))
	}
	if len(m.X) == 0 || len(m.X)%f.NFeatures != 0 {
		return errMsg(m.ReqID, fmt.Errorf("X debe tener un múltiplo de %d valores", f.NFeatures))
	}
	return proto.Msg{
		Kind: proto.PredictResp, ReqID: m.ReqID,
		Probs: f.PredictBatch(m.X, n.Workers), NTrees: len(f.Trees),
	}
}
