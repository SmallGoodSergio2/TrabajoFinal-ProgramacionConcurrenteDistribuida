package dataset

import (
	"hash/fnv"
	"math"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"sync"
)

// Índices de las características que recibe el modelo.
const (
	FeatAnio = iota
	FeatMes
	FeatDpto
	FeatProv
	FeatUbigeo
	FeatVolumen
	NumFeatures
)

var FeatureNames = []string{"anio", "mes", "departamento", "provincia", "ubigeo", "volumen_distrital"}

// District agrupa la información de un distrito calculada sobre el set de entrenamiento.
type District struct {
	Ubigeo     int     `json:"ubigeo"`
	Dpto       string  `json:"dpto"`
	Prov       string  `json:"prov"`
	Dist       string  `json:"dist"`
	Total      int64   `json:"total"`
	Months     int     `json:"months"`
	AvgMonthly float64 `json:"avg_monthly"`
}

// MonthKey identifica las denuncias de un distrito en un mes concreto.
type MonthKey struct {
	Ubigeo, Anio, Mes int
}

// Prepared es el dataset listo para entrenar, con los índices de
// entrenamiento/prueba, los distritos y los agregados.
type Prepared struct {
	Records    []Record
	Classes    []string
	ClassIndex map[string]int
	Districts  map[int]*District
	// DistrictList está ordenada por ubigeo.
	DistrictList []*District
	Train        []int
	Test         []int
	Seasonality  [12]float64
	Aggregates   *Aggregates
	Load         LoadStats
	MinYear      int
	MaxYear      int
}

// Prepare construye las estructuras derivadas usando varias goroutines.
func Prepare(records []Record, load LoadStats, testFraction float64, workers int) *Prepared {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	p := &Prepared{Records: records, Load: load, Districts: map[int]*District{}, MinYear: math.MaxInt}

	classSet := map[string]struct{}{}
	for i := range records {
		r := &records[i]
		classSet[r.Modalidad] = struct{}{}
		p.MinYear = min(p.MinYear, r.Anio)
		p.MaxYear = max(p.MaxYear, r.Anio)
	}
	for c := range classSet {
		p.Classes = append(p.Classes, c)
	}
	sort.Strings(p.Classes)
	p.ClassIndex = make(map[string]int, len(p.Classes))
	for i, c := range p.Classes {
		p.ClassIndex[c] = i
	}

	threshold := uint32(testFraction * 1000)
	monthsSeen := map[int]map[int]struct{}{}
	for i := range records {
		r := &records[i]
		key := MonthKey{r.Ubigeo, r.Anio, r.Mes}
		d := p.Districts[r.Ubigeo]
		if d == nil {
			d = &District{Ubigeo: r.Ubigeo, Dpto: r.Dpto, Prov: r.Prov, Dist: r.Dist}
			p.Districts[r.Ubigeo] = d
			monthsSeen[r.Ubigeo] = map[int]struct{}{}
		}
		if r.Cantidad == 0 {
			continue
		}
		if groupHash(key)%1000 < threshold {
			p.Test = append(p.Test, i)
			continue
		}
		p.Train = append(p.Train, i)
		d.Total += int64(r.Cantidad)
		monthsSeen[r.Ubigeo][r.Anio*12+r.Mes] = struct{}{}
	}
	for u, d := range p.Districts {
		d.Months = len(monthsSeen[u])
		if d.Months > 0 {
			d.AvgMonthly = float64(d.Total) / float64(d.Months)
		}
		p.DistrictList = append(p.DistrictList, d)
	}
	slices.SortFunc(p.DistrictList, func(a, b *District) int { return a.Ubigeo - b.Ubigeo })

	p.Aggregates = aggregate(records, p.Districts, workers)
	p.Seasonality = seasonality(records)
	return p
}

// seasonality devuelve el índice mensual (1 = mes promedio) usando solo los
// años con los 12 meses registrados, para que un año parcial no sesgue los
// meses que le faltan.
func seasonality(records []Record) [12]float64 {
	byYear := map[int]*[12]int64{}
	for i := range records {
		r := &records[i]
		m := byYear[r.Anio]
		if m == nil {
			m = &[12]int64{}
			byYear[r.Anio] = m
		}
		m[r.Mes-1] += int64(r.Cantidad)
	}
	var totals [12]float64
	for _, months := range byYear {
		if slices.Contains(months[:], 0) {
			continue
		}
		for i, v := range months {
			totals[i] += float64(v)
		}
	}
	var mean float64
	for _, v := range totals {
		mean += v / 12
	}
	var idx [12]float64
	for i, v := range totals {
		idx[i] = 1
		if mean > 0 {
			idx[i] = v / mean
		}
	}
	return idx
}

func groupHash(k MonthKey) uint32 {
	h := fnv.New32a()
	var buf [32]byte
	b := strconv.AppendInt(buf[:0], int64(k.Ubigeo), 10)
	b = append(b, '|')
	b = strconv.AppendInt(b, int64(k.Anio), 10)
	b = append(b, '|')
	b = strconv.AppendInt(b, int64(k.Mes), 10)
	h.Write(b)
	return h.Sum32()
}

// Features escribe el vector de características para un distrito y mes.
func (p *Prepared) Features(ubigeo, anio, mes int, dst []float32) ([]float32, bool) {
	d, ok := p.Districts[ubigeo]
	if !ok {
		return dst, false
	}
	return append(dst,
		float32(anio),
		float32(mes),
		float32(ubigeo/10000),
		float32(ubigeo/100),
		float32(ubigeo),
		float32(math.Log1p(d.AvgMonthly)),
	), true
}

// Matrix genera la matriz de entrenamiento (fila mayor) para los índices dados.
// Cada fila agregada pesa tanto como su número de denuncias.
func (p *Prepared) Matrix(idx []int) (x []float32, y []uint8, w []float32) {
	x = make([]float32, 0, len(idx)*NumFeatures)
	y = make([]uint8, len(idx))
	w = make([]float32, len(idx))
	for j, i := range idx {
		r := &p.Records[i]
		x, _ = p.Features(r.Ubigeo, r.Anio, r.Mes, x)
		y[j] = uint8(p.ClassIndex[r.Modalidad])
		w[j] = float32(r.Cantidad)
	}
	return x, y, w
}

// Aggregates contiene estadísticas descriptivas para el panel de impacto social.
type Aggregates struct {
	TotalDenuncias   int64                       `json:"total_denuncias"`
	Registros        int                         `json:"registros"`
	PorModalidad     map[string]int64            `json:"por_modalidad"`
	PorAnio          map[int]int64               `json:"por_anio"`
	PorMes           [12]int64                   `json:"por_mes"`
	PorDpto          map[string]int64            `json:"por_dpto"`
	PorDptoModalidad map[string]map[string]int64 `json:"por_dpto_modalidad"`
	TopDistritos     []DistrictTotal             `json:"top_distritos"`
}

type DistrictTotal struct {
	Ubigeo int    `json:"ubigeo"`
	Dpto   string `json:"dpto"`
	Prov   string `json:"prov"`
	Dist   string `json:"dist"`
	Total  int64  `json:"total"`
}

type partial struct {
	total        int64
	porModalidad map[string]int64
	porAnio      map[int]int64
	porMes       [12]int64
	porDpto      map[string]int64
	porDptoMod   map[string]map[string]int64
	porDistrito  map[int]int64
}

// aggregate aplica map-reduce: cada goroutine resume un segmento de los
// registros y luego se combinan los parciales.
func aggregate(records []Record, districts map[int]*District, workers int) *Aggregates {
	parts := make([]partial, workers)
	var wg sync.WaitGroup
	step := (len(records) + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo, hi := w*step, min((w+1)*step, len(records))
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(pt *partial, seg []Record) {
			defer wg.Done()
			pt.porModalidad = map[string]int64{}
			pt.porAnio = map[int]int64{}
			pt.porDpto = map[string]int64{}
			pt.porDptoMod = map[string]map[string]int64{}
			pt.porDistrito = map[int]int64{}
			for i := range seg {
				r := &seg[i]
				c := int64(r.Cantidad)
				pt.total += c
				pt.porModalidad[r.Modalidad] += c
				pt.porAnio[r.Anio] += c
				pt.porMes[r.Mes-1] += c
				pt.porDpto[r.Dpto] += c
				m := pt.porDptoMod[r.Dpto]
				if m == nil {
					m = map[string]int64{}
					pt.porDptoMod[r.Dpto] = m
				}
				m[r.Modalidad] += c
				pt.porDistrito[r.Ubigeo] += c
			}
		}(&parts[w], records[lo:hi])
	}
	wg.Wait()

	agg := &Aggregates{
		Registros:        len(records),
		PorModalidad:     map[string]int64{},
		PorAnio:          map[int]int64{},
		PorDpto:          map[string]int64{},
		PorDptoModalidad: map[string]map[string]int64{},
	}
	porDistrito := map[int]int64{}
	for _, pt := range parts {
		agg.TotalDenuncias += pt.total
		for k, v := range pt.porModalidad {
			agg.PorModalidad[k] += v
		}
		for k, v := range pt.porAnio {
			agg.PorAnio[k] += v
		}
		for i, v := range pt.porMes {
			agg.PorMes[i] += v
		}
		for k, v := range pt.porDpto {
			agg.PorDpto[k] += v
		}
		for dpto, mods := range pt.porDptoMod {
			m := agg.PorDptoModalidad[dpto]
			if m == nil {
				m = map[string]int64{}
				agg.PorDptoModalidad[dpto] = m
			}
			for k, v := range mods {
				m[k] += v
			}
		}
		for k, v := range pt.porDistrito {
			porDistrito[k] += v
		}
	}
	for u, total := range porDistrito {
		d := districts[u]
		agg.TopDistritos = append(agg.TopDistritos, DistrictTotal{Ubigeo: u, Dpto: d.Dpto, Prov: d.Prov, Dist: d.Dist, Total: total})
	}
	slices.SortFunc(agg.TopDistritos, func(a, b DistrictTotal) int {
		if a.Total != b.Total {
			if a.Total > b.Total {
				return -1
			}
			return 1
		}
		return a.Ubigeo - b.Ubigeo
	})
	if len(agg.TopDistritos) > 20 {
		agg.TopDistritos = agg.TopDistritos[:20]
	}
	return agg
}
