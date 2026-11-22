package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	dbName             = "cc65_rf"
	colPredictions     = "predictions"
	colTrainRuns       = "train_runs"
	redisPredictTTL    = 60 * time.Second
	redisPredictPrefix = "predict:"
)

type Prediction struct {
	ID        primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	Ubigeo    int                `json:"ubigeo" bson:"ubigeo"`
	Anio      int                `json:"anio" bson:"anio"`
	Mes       int                `json:"mes" bson:"mes"`
	TopClass  string             `json:"top_class" bson:"top_class"`
	Probs     map[string]float64 `json:"probs" bson:"probs"`
	LatencyMs float64            `json:"latency_ms" bson:"latency_ms"`
	Nodes     int                `json:"nodes" bson:"nodes"`
	Trees     int                `json:"trees" bson:"trees"`
	Cached    bool               `json:"cached" bson:"cached"`
	CreatedAt time.Time          `json:"created_at" bson:"created_at"`
}

type TrainNodeResult struct {
	NodeID string  `json:"node_id" bson:"node_id"`
	Shard  int     `json:"shard" bson:"shard"`
	Rows   int     `json:"rows" bson:"rows"`
	Trees  int     `json:"trees" bson:"trees"`
	Millis float64 `json:"millis" bson:"millis"`
	Error  string  `json:"error,omitempty" bson:"error,omitempty"`
}

type TrainRun struct {
	ID         primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	TreesTotal int                `json:"trees_total" bson:"trees_total"`
	Nodes      int                `json:"nodes" bson:"nodes"`
	DurationMs float64            `json:"duration_ms" bson:"duration_ms"`
	DataPath   string             `json:"data_path" bson:"data_path"`
	Results    []TrainNodeResult  `json:"results" bson:"results"`
	CreatedAt  time.Time          `json:"created_at" bson:"created_at"`
}

type Store struct {
	mongo *mongo.Client
	db    *mongo.Database
	rdb   *redis.Client
}

func Open(ctx context.Context, mongoURI, redisAddr string) (*Store, error) {
	s := &Store{}
	if mongoURI != "" {
		cli, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoURI))
		if err != nil {
			return nil, fmt.Errorf("mongo: %w", err)
		}
		if err := cli.Ping(ctx, nil); err != nil {
			_ = cli.Disconnect(ctx)
			return nil, fmt.Errorf("mongo ping: %w", err)
		}
		s.mongo = cli
		s.db = cli.Database(dbName)
		_, _ = s.db.Collection(colPredictions).Indexes().CreateOne(ctx, mongo.IndexModel{
			Keys: bson.D{{Key: "created_at", Value: -1}},
		})
		_, _ = s.db.Collection(colTrainRuns).Indexes().CreateOne(ctx, mongo.IndexModel{
			Keys: bson.D{{Key: "created_at", Value: -1}},
		})
	}
	if redisAddr != "" {
		rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
		if err := rdb.Ping(ctx).Err(); err != nil {
			_ = rdb.Close()
			if s.mongo != nil {
				_ = s.mongo.Disconnect(ctx)
			}
			return nil, fmt.Errorf("redis ping: %w", err)
		}
		s.rdb = rdb
	}
	return s, nil
}

func (s *Store) Close(ctx context.Context) error {
	var err error
	if s.rdb != nil {
		err = s.rdb.Close()
	}
	if s.mongo != nil {
		if e := s.mongo.Disconnect(ctx); e != nil && err == nil {
			err = e
		}
	}
	return err
}

func (s *Store) SavePrediction(ctx context.Context, p *Prediction) error {
	if s == nil || s.db == nil {
		return nil
	}
	if p.CreatedAt.IsZero() {
		p.CreatedAt = time.Now().UTC()
	}
	res, err := s.db.Collection(colPredictions).InsertOne(ctx, p)
	if err != nil {
		return err
	}
	if id, ok := res.InsertedID.(primitive.ObjectID); ok {
		p.ID = id
	}
	return nil
}

func (s *Store) ListPredictions(ctx context.Context, limit int64) ([]Prediction, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 50
	}
	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: -1}}).SetLimit(limit)
	cur, err := s.db.Collection(colPredictions).Find(ctx, bson.D{}, opts)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []Prediction
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *Store) SaveTrainRun(ctx context.Context, t *TrainRun) error {
	if s == nil || s.db == nil {
		return nil
	}
	if t.CreatedAt.IsZero() {
		t.CreatedAt = time.Now().UTC()
	}
	res, err := s.db.Collection(colTrainRuns).InsertOne(ctx, t)
	if err != nil {
		return err
	}
	if id, ok := res.InsertedID.(primitive.ObjectID); ok {
		t.ID = id
	}
	return nil
}

func (s *Store) ListTrainRuns(ctx context.Context, limit int64) ([]TrainRun, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 20
	}
	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: -1}}).SetLimit(limit)
	cur, err := s.db.Collection(colTrainRuns).Find(ctx, bson.D{}, opts)
	if err != nil {
		return nil, err
	}
	defer cur.Close(ctx)
	var out []TrainRun
	if err := cur.All(ctx, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func cacheKey(ubigeo, anio, mes int) string {
	return fmt.Sprintf("%s%d:%d:%d", redisPredictPrefix, ubigeo, anio, mes)
}

func (s *Store) GetCachedPrediction(ctx context.Context, ubigeo, anio, mes int) (*Prediction, bool) {
	if s == nil || s.rdb == nil {
		return nil, false
	}
	raw, err := s.rdb.Get(ctx, cacheKey(ubigeo, anio, mes)).Bytes()
	if err != nil {
		return nil, false
	}
	var p Prediction
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, false
	}
	p.Cached = true
	return &p, true
}

func (s *Store) CachePrediction(ctx context.Context, p *Prediction) error {
	if s == nil || s.rdb == nil || p == nil {
		return nil
	}
	raw, err := json.Marshal(p)
	if err != nil {
		return err
	}
	return s.rdb.Set(ctx, cacheKey(p.Ubigeo, p.Anio, p.Mes), raw, redisPredictTTL).Err()
}

func (s *Store) IncrPredictCounter(ctx context.Context) (int64, error) {
	if s == nil || s.rdb == nil {
		return 0, nil
	}
	key := "metrics:predict:" + time.Now().UTC().Format("2006-01-02")
	n, err := s.rdb.Incr(ctx, key).Result()
	if err != nil {
		return 0, err
	}
	_ = s.rdb.Expire(ctx, key, 48*time.Hour).Err()
	return n, nil
}
