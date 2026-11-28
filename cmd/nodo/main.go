package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/SmallGoodSergio2/TrabajoFinal-ProgramacionConcurrenteDistribuida/internal/node"
)

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	host, _ := os.Hostname() // en Docker, el id del contenedor: único por réplica
	coord := flag.String("coord", getenv("COORD_ADDR", "localhost:9000"), "dirección host:puerto del coordinador")
	id := flag.String("id", getenv("NODE_ID", host), "identificador único del nodo")
	workers := flag.Int("workers", 0, "goroutines para cargar y entrenar (0 = núcleos disponibles)")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	n := &node.Node{ID: *id, Coord: *coord, Workers: *workers}
	log.Printf("nodo %s -> coordinador %s", n.ID, n.Coord)
	_ = n.Run(ctx)
}
