// Package coordinator implementa el centro de la topología en estrella: acepta
// las conexiones TCP de los nodos, reparte el entrenamiento y reúne las
// predicciones parciales. No sabe nada de HTTP; el API lo usa como biblioteca.
package coordinator

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"log"
	"net"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/proto"
	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"
)

const (
	pingEvery = 2 * time.Second
	// Un nodo que no envía nada (ni Pong) en este tiempo se considera caído.
	nodeTimeout = 10 * time.Second
	// Plazo por defecto de una predicción si el llamador no puso uno.
	DefaultPredictTimeout = 80 * time.Millisecond
)

var ErrNoNodes = errors.New("no hay nodos conectados")

// Hub es el coordinador. Es seguro para uso concurrente.
type Hub struct {
	ln      net.Listener
	mu      sync.RWMutex
	nodes   map[string]*nodeConn
	nextReq atomic.Uint64
}

// Listen abre el puerto TCP donde los nodos se registran.
func Listen(addr string) (*Hub, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, err
	}
	h := &Hub{ln: ln, nodes: map[string]*nodeConn{}}
	go h.acceptLoop()
	return h, nil
}

func (h *Hub) Addr() net.Addr { return h.ln.Addr() }

func (h *Hub) Close() error {
	err := h.ln.Close()
	for _, n := range h.snapshot() {
		n.close()
	}
	return err
}

func (h *Hub) acceptLoop() {
	for {
		c, err := h.ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return
			}
			log.Printf("[coord] accept: %v", err)
			time.Sleep(50 * time.Millisecond)
			continue
		}
		go h.handshake(c)
	}
}

// handshake espera el mensaje Register y, si es válido, da de alta al nodo.
func (h *Hub) handshake(c net.Conn) {
	_ = c.SetReadDeadline(time.Now().Add(5 * time.Second))
	dec := gob.NewDecoder(c)
	var m proto.Msg
	if err := dec.Decode(&m); err != nil || m.Kind != proto.Register || m.NodeID == "" {
		c.Close()
		return
	}
	_ = c.SetReadDeadline(time.Time{})

	n := &nodeConn{
		id: m.NodeID, addr: c.RemoteAddr().String(), cores: m.Cores,
		conn: c, enc: gob.NewEncoder(c), dec: dec,
		pending: map[uint64]chan proto.Msg{},
		done:    make(chan struct{}),
		hub:     h,
	}
	n.lastSeen.Store(time.Now().UnixNano())

	h.mu.Lock()
	old := h.nodes[n.id]
	h.nodes[n.id] = n
	h.mu.Unlock()
	if old != nil { // el mismo nodo se reconectó: cerrar la sesión anterior
		old.close()
	}
	log.Printf("[coord] nodo %s conectado desde %s (%d núcleos). Nodos activos: %d", n.id, n.addr, n.cores, h.NodeCount())

	go n.readLoop()
	go n.pingLoop()
}

func (h *Hub) remove(n *nodeConn) {
	h.mu.Lock()
	if h.nodes[n.id] == n { // no borrar una sesión más nueva del mismo nodo
		delete(h.nodes, n.id)
	}
	h.mu.Unlock()
	log.Printf("[coord] nodo %s desconectado. Nodos activos: %d", n.id, h.NodeCount())
}

func (h *Hub) snapshot() []*nodeConn {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]*nodeConn, 0, len(h.nodes))
	for _, n := range h.nodes {
		out = append(out, n)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

func (h *Hub) NodeCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes)
}

// WaitForNodes bloquea hasta que haya al menos n nodos conectados.
func (h *Hub) WaitForNodes(ctx context.Context, n int) error {
	t := time.NewTicker(100 * time.Millisecond)
	defer t.Stop()
	for {
		if h.NodeCount() >= n {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("esperando %d nodos (conectados %d): %w", n, h.NodeCount(), ctx.Err())
		case <-t.C:
		}
	}
}

// ---------------------------------------------------------------- Entrenar

type TrainConfig struct {
	Params       rf.Params // NumTrees es el total de árboles del bosque completo
	DataPath     string    // ruta del CSV dentro del nodo
	TestFraction float64
}

// NodeTrain es el resultado del entrenamiento en un nodo.
type NodeTrain struct {
	NodeID string
	Shard  int
	Rows   int
	Trees  int
	Millis float64
	Err    error
}

// Train reparte NumTrees entre los nodos conectados y espera a que todos
// terminen (fork-join). Cada nodo recibe un shard distinto de los datos y una
// semilla distinta. Si un nodo falla, se devuelve su error pero los demás
// siguen siendo válidos.
func (h *Hub) Train(ctx context.Context, cfg TrainConfig) ([]NodeTrain, error) {
	nodes := h.snapshot()
	if len(nodes) == 0 {
		return nil, ErrNoNodes
	}
	if cfg.Params.NumTrees <= 0 {
		cfg.Params.NumTrees = 30
	}
	if cfg.Params.Seed == 0 {
		cfg.Params.Seed = 42
	}
	n := len(nodes)
	base, extra := cfg.Params.NumTrees/n, cfg.Params.NumTrees%n

	out := make([]NodeTrain, n) // cada goroutine escribe solo en su posición
	var wg sync.WaitGroup
	for i, nd := range nodes {
		trees := base
		if i < extra {
			trees++
		}
		out[i] = NodeTrain{NodeID: nd.id, Shard: i}
		if trees == 0 {
			continue
		}
		wg.Add(1)
		go func(i int, nd *nodeConn, trees int) {
			defer wg.Done()
			p := cfg.Params
			p.NumTrees = trees
			p.Seed = cfg.Params.Seed + uint64(i)*104729
			r, err := nd.call(ctx, proto.Msg{
				Kind: proto.Train, Params: p, DataPath: cfg.DataPath,
				TestFraction: cfg.TestFraction, Shard: i, NShards: n,
			})
			if err != nil {
				out[i].Err = err
				return
			}
			out[i].Rows, out[i].Trees, out[i].Millis = r.Rows, r.NTrees, r.Millis
			nd.trees.Store(int32(r.NTrees))
		}(i, nd, trees)
	}
	wg.Wait()

	var errs []error
	for _, r := range out {
		if r.Err != nil {
			errs = append(errs, r.Err)
		}
	}
	return out, errors.Join(errs...)
}

// --------------------------------------------------------------- Predecir

type PredictResult struct {
	Probs []float32 // distribución de clases normalizada (n filas * NClasses)
	Nodes int       // nodos que respondieron a tiempo
	Trees int       // árboles que votaron
}

// Predict hace fan-out de las filas x a todos los nodos, espera sus sumas
// parciales (fan-in) y normaliza entre los árboles que SÍ respondieron. Si un
// nodo cae o es lento, la predicción se degrada pero no falla.
func (h *Hub) Predict(ctx context.Context, x []float32) (PredictResult, error) {
	nodes := h.snapshot()
	if len(nodes) == 0 {
		return PredictResult{}, ErrNoNodes
	}
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, DefaultPredictTimeout)
		defer cancel()
	}

	type reply struct {
		m   proto.Msg
		err error
	}
	ch := make(chan reply, len(nodes)) // con búfer: ninguna goroutine queda bloqueada
	for _, nd := range nodes {
		go func(nd *nodeConn) {
			m, err := nd.call(ctx, proto.Msg{Kind: proto.Predict, X: x})
			ch <- reply{m, err}
		}(nd)
	}

	var res PredictResult
	var sum []float32
	var lastErr error
	for range nodes {
		r := <-ch
		if r.err != nil {
			lastErr = r.err
			continue
		}
		if sum == nil {
			sum = make([]float32, len(r.m.Probs))
		}
		if len(r.m.Probs) != len(sum) {
			lastErr = fmt.Errorf("nodo respondió con %d valores, se esperaban %d", len(r.m.Probs), len(sum))
			continue
		}
		for k, v := range r.m.Probs {
			sum[k] += v
		}
		res.Trees += r.m.NTrees
		res.Nodes++
	}
	if res.Trees == 0 {
		return res, fmt.Errorf("ningún nodo respondió: %w", lastErr)
	}
	inv := 1 / float32(res.Trees)
	for k := range sum {
		sum[k] *= inv
	}
	res.Probs = sum
	return res, nil
}

// ------------------------------------------------------------------ Estado

// NodeInfo alimenta el panel de administrador (nodos, latencia, árboles).
type NodeInfo struct {
	ID         string  `json:"id"`
	Addr       string  `json:"addr"`
	Cores      int     `json:"cores"`
	Trees      int     `json:"trees"`
	LastSeenMs int64   `json:"last_seen_ms"`
	RTTMs      float64 `json:"rtt_ms"`
}

func (h *Hub) Status() []NodeInfo {
	nodes := h.snapshot()
	out := make([]NodeInfo, len(nodes))
	for i, n := range nodes {
		out[i] = NodeInfo{
			ID: n.id, Addr: n.addr, Cores: n.cores, Trees: int(n.trees.Load()),
			LastSeenMs: time.Since(time.Unix(0, n.lastSeen.Load())).Milliseconds(),
			RTTMs:      float64(n.rttNs.Load()) / 1e6,
		}
	}
	return out
}

// -------------------------------------------------------------- Conexión

// nodeConn es la conexión con un nodo. Varias goroutines pueden llamar a
// call() a la vez: las escrituras se serializan con wmu y las respuestas se
// entregan por ReqID a través del mapa pending.
type nodeConn struct {
	id, addr string
	cores    int
	conn     net.Conn
	enc      *gob.Encoder
	dec      *gob.Decoder
	hub      *Hub

	wmu sync.Mutex // protege enc

	pmu     sync.Mutex // protege pending
	pending map[uint64]chan proto.Msg

	done chan struct{}
	once sync.Once

	lastSeen atomic.Int64 // UnixNano del último mensaje recibido
	pingSent atomic.Int64
	rttNs    atomic.Int64
	trees    atomic.Int32
}

func (n *nodeConn) close() {
	n.once.Do(func() {
		close(n.done)
		n.conn.Close()
	})
}

func (n *nodeConn) send(m proto.Msg) error {
	n.wmu.Lock()
	defer n.wmu.Unlock()
	_ = n.conn.SetWriteDeadline(time.Now().Add(5 * time.Second))
	if err := n.enc.Encode(m); err != nil {
		n.close() // un Encode a medias deja el flujo gob inservible
		return err
	}
	return nil
}

func (n *nodeConn) readLoop() {
	defer n.hub.remove(n)
	defer n.close()
	for {
		var m proto.Msg // nuevo en cada vuelta (gob no limpia campos omitidos)
		if err := n.dec.Decode(&m); err != nil {
			return
		}
		n.lastSeen.Store(time.Now().UnixNano())
		if m.Kind == proto.Pong {
			if t := n.pingSent.Load(); t != 0 {
				n.rttNs.Store(time.Now().UnixNano() - t)
			}
			continue
		}
		n.pmu.Lock()
		ch := n.pending[m.ReqID]
		delete(n.pending, m.ReqID)
		n.pmu.Unlock()
		if ch != nil {
			ch <- m // búfer de 1: nunca bloquea
		}
	}
}

func (n *nodeConn) pingLoop() {
	t := time.NewTicker(pingEvery)
	defer t.Stop()
	for {
		select {
		case <-n.done:
			return
		case <-t.C:
			if time.Since(time.Unix(0, n.lastSeen.Load())) > nodeTimeout {
				log.Printf("[coord] nodo %s no responde: se desconecta", n.id)
				n.close()
				return
			}
			n.pingSent.Store(time.Now().UnixNano())
			if err := n.send(proto.Msg{Kind: proto.Ping}); err != nil {
				return
			}
		}
	}
}

// call envía una petición y espera su respuesta, el plazo del contexto o la
// caída del nodo, lo que ocurra primero.
func (n *nodeConn) call(ctx context.Context, m proto.Msg) (proto.Msg, error) {
	id := n.hub.nextReq.Add(1)
	m.ReqID = id
	ch := make(chan proto.Msg, 1)
	n.pmu.Lock()
	n.pending[id] = ch
	n.pmu.Unlock()
	defer func() {
		n.pmu.Lock()
		delete(n.pending, id)
		n.pmu.Unlock()
	}()

	if err := n.send(m); err != nil {
		return proto.Msg{}, fmt.Errorf("nodo %s: %w", n.id, err)
	}
	select {
	case r := <-ch:
		if r.Kind == proto.Err {
			return r, fmt.Errorf("nodo %s: %s", n.id, r.Err)
		}
		return r, nil
	case <-ctx.Done():
		return proto.Msg{}, fmt.Errorf("nodo %s: %w", n.id, ctx.Err())
	case <-n.done:
		return proto.Msg{}, fmt.Errorf("nodo %s desconectado", n.id)
	}
}
