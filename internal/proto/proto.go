package proto

import "github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/rf"

type Kind uint8

const (
	Register    Kind = iota + 1 // nodo -> coord: se presenta (NodeID, Cores)
	Train                       // coord -> nodo: entrena tu shard
	TrainDone                   // nodo -> coord: terminé (NTrees, Rows, Millis)
	Predict                     // coord -> nodo: predice las filas de X
	PredictResp                 // nodo -> coord: suma de distribuciones (Probs)
	Ping                        // coord -> nodo: latido
	Pong                        // nodo -> coord: respuesta al latido
	Err                         // nodo -> coord: falló la petición ReqID
)

func (k Kind) String() string {
	switch k {
	case Register:
		return "REGISTER"
	case Train:
		return "TRAIN"
	case TrainDone:
		return "TRAIN_DONE"
	case Predict:
		return "PREDICT"
	case PredictResp:
		return "PREDICT_RESP"
	case Ping:
		return "PING"
	case Pong:
		return "PONG"
	case Err:
		return "ERR"
	}
	return "?"
}

type Msg struct {
	Kind  Kind
	ReqID uint64

	// Register
	NodeID string
	Cores  int

	// Train
	Params       rf.Params
	DataPath     string
	TestFraction float64
	Shard        int
	NShards      int

	// TrainDone
	Rows   int
	Millis float64

	// Predict (X, fila-mayor) y PredictResp (Probs: suma de los árboles del nodo)
	X      []float32
	Probs  []float32
	NTrees int

	// Err
	Err string
}
