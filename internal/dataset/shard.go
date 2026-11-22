package dataset

import (
	"hash/fnv"
	"strconv"
)

func ShardOf(k MonthKey, n int) int {
	if n <= 1 {
		return 0
	}
	h := fnv.New32a()
	h.Write([]byte("shard|"))
	var buf [32]byte
	b := strconv.AppendInt(buf[:0], int64(k.Ubigeo), 10)
	b = append(b, '|')
	b = strconv.AppendInt(b, int64(k.Anio), 10)
	b = append(b, '|')
	b = strconv.AppendInt(b, int64(k.Mes), 10)
	h.Write(b)
	return int(h.Sum32() % uint32(n))
}
