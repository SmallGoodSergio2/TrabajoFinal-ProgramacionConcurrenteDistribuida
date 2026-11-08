// Package dataset implementa la Parte A: carga concurrente, limpieza y
// validación del dataset de denuncias policiales (MININTER - PNP).
package dataset

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Record es una fila limpia y validada del dataset.
type Record struct {
	Anio      int
	Mes       int
	Dpto      string
	Prov      string
	Dist      string
	Ubigeo    int
	Modalidad string
	Cantidad  int
}

// LoadStats resume la ejecución del cargador concurrente.
type LoadStats struct {
	File       string  `json:"file" bson:"file"`
	Bytes      int64   `json:"bytes" bson:"bytes"`
	Workers    int     `json:"workers" bson:"workers"`
	Chunks     int     `json:"chunks" bson:"chunks"`
	Lines      int     `json:"lines" bson:"lines"`
	Valid      int     `json:"valid" bson:"valid"`
	Invalid    int     `json:"invalid" bson:"invalid"`
	Duplicates int     `json:"duplicates" bson:"duplicates"`
	DurationMs float64 `json:"duration_ms" bson:"duration_ms"`
	RowsPerSec float64 `json:"rows_per_sec" bson:"rows_per_sec"`
}

const (
	colAnio = iota
	colMes
	colDpto
	colProv
	colDist
	colUbigeo
	colModalidad
	colCantidad
	numColumns
)

var columnNames = [numColumns]string{
	"ANIO", "MES", "DPTO_HECHO_NEW", "PROV_HECHO", "DIST_HECHO",
	"UBIGEO_HECHO", "P_MODALIDADES", "cantidad",
}

const (
	minYear = 2015
	maxYear = 2026
)

type chunk struct {
	index      int
	start, end int64
}

type chunkResult struct {
	index   int
	records []Record
	lines   int
	invalid int
	err     error
}

// LoadCSV lee el archivo dividiéndolo en bloques de bytes que se procesan
// en paralelo por un pool de goroutines. Los resultados se reensamblan en
// el orden original y se eliminan los duplicados exactos.
func LoadCSV(path string, workers int) ([]Record, LoadStats, error) {
	begin := time.Now()
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	stats := LoadStats{File: path, Workers: workers}

	f, err := os.Open(path)
	if err != nil {
		return nil, stats, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, stats, err
	}
	size := info.Size()
	stats.Bytes = size

	cols, headerEnd, err := readHeader(f, size)
	if err != nil {
		return nil, stats, err
	}

	chunks := splitChunks(headerEnd, size, workers*4)
	stats.Chunks = len(chunks)

	jobs := make(chan chunk)
	results := make(chan chunkResult, len(chunks))
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p := newParser(cols)
			for c := range jobs {
				results <- p.parseChunk(f, size, c)
			}
		}()
	}
	go func() {
		for _, c := range chunks {
			jobs <- c
		}
		close(jobs)
		wg.Wait()
		close(results)
	}()

	parts := make([][]Record, len(chunks))
	var firstErr error
	total := 0
	for r := range results {
		if r.err != nil && firstErr == nil {
			firstErr = r.err
		}
		parts[r.index] = r.records
		stats.Lines += r.lines
		stats.Invalid += r.invalid
		total += len(r.records)
	}
	if firstErr != nil {
		return nil, stats, firstErr
	}

	seen := make(map[Record]struct{}, total)
	out := make([]Record, 0, total)
	for _, part := range parts {
		for _, rec := range part {
			if _, dup := seen[rec]; dup {
				stats.Duplicates++
				continue
			}
			seen[rec] = struct{}{}
			out = append(out, rec)
		}
	}
	stats.Valid = len(out)
	elapsed := time.Since(begin)
	stats.DurationMs = float64(elapsed.Microseconds()) / 1000
	if elapsed > 0 {
		stats.RowsPerSec = float64(stats.Lines) / elapsed.Seconds()
	}
	return out, stats, nil
}

func splitChunks(start, size int64, n int) []chunk {
	body := size - start
	if body <= 0 {
		return nil
	}
	const minChunk = 64 << 10
	if int64(n) > body/minChunk {
		n = int(body / minChunk)
	}
	if n < 1 {
		n = 1
	}
	step := body / int64(n)
	chunks := make([]chunk, n)
	for i := range chunks {
		chunks[i] = chunk{index: i, start: start + int64(i)*step, end: start + int64(i+1)*step}
	}
	chunks[n-1].end = size
	return chunks
}

func readHeader(f *os.File, size int64) ([numColumns]int, int64, error) {
	var cols [numColumns]int
	br := bufio.NewReader(io.NewSectionReader(f, 0, size))
	line, err := br.ReadBytes('\n')
	if err != nil && !errors.Is(err, io.EOF) {
		return cols, 0, err
	}
	headerEnd := int64(len(line))
	line = bytes.TrimPrefix(line, []byte("\xef\xbb\xbf"))
	fields := splitFields(trimEOL(line), nil)
	index := make(map[string]int, len(fields))
	for i, fld := range fields {
		index[strings.TrimSpace(string(fld))] = i
	}
	for c, name := range columnNames {
		i, ok := index[name]
		if !ok {
			return cols, 0, fmt.Errorf("columna requerida %q no encontrada en la cabecera", name)
		}
		cols[c] = i
	}
	return cols, headerEnd, nil
}

type parser struct {
	cols    [numColumns]int
	maxCol  int
	fields  [][]byte
	strings map[string]string
}

func newParser(cols [numColumns]int) *parser {
	maxCol := 0
	for _, c := range cols {
		maxCol = max(maxCol, c)
	}
	return &parser{cols: cols, maxCol: maxCol, strings: make(map[string]string, 4096)}
}

// parseChunk procesa las líneas que empiezan dentro de [c.start, c.end).
// Si el bloque no empieza en un inicio de línea, la línea parcial pertenece
// al bloque anterior y se descarta.
func (p *parser) parseChunk(f *os.File, size int64, c chunk) chunkResult {
	res := chunkResult{index: c.index}
	offset := c.start
	skipPartial := c.index > 0
	if skipPartial {
		offset--
	}
	br := bufio.NewReaderSize(io.NewSectionReader(f, offset, size-offset), 1<<20)
	pos := offset
	if skipPartial {
		line, err := br.ReadSlice('\n')
		pos += int64(len(line))
		if err != nil {
			if !errors.Is(err, io.EOF) {
				res.err = err
			}
			return res
		}
	}
	res.records = make([]Record, 0, (c.end-c.start)/48)
	for pos < c.end {
		line, err := br.ReadSlice('\n')
		pos += int64(len(line))
		if err != nil && !errors.Is(err, io.EOF) {
			res.err = fmt.Errorf("bloque %d: %w", c.index, err)
			return res
		}
		if line = trimEOL(line); len(line) > 0 {
			res.lines++
			if rec, ok := p.parseLine(line); ok {
				res.records = append(res.records, rec)
			} else {
				res.invalid++
			}
		}
		if err != nil {
			break
		}
	}
	return res
}

func (p *parser) parseLine(line []byte) (Record, bool) {
	p.fields = splitFields(line, p.fields)
	if len(p.fields) <= p.maxCol {
		return Record{}, false
	}
	get := func(c int) []byte { return p.fields[p.cols[c]] }

	anio, ok1 := parseInt(get(colAnio))
	mes, ok2 := parseInt(get(colMes))
	cantidad, ok3 := parseInt(get(colCantidad))
	ubigeo, ok4 := parseUbigeo(get(colUbigeo))
	if !ok1 || !ok2 || !ok3 || !ok4 ||
		anio < minYear || anio > maxYear || mes < 1 || mes > 12 || cantidad < 0 {
		return Record{}, false
	}
	rec := Record{
		Anio:      anio,
		Mes:       mes,
		Ubigeo:    ubigeo,
		Cantidad:  cantidad,
		Dpto:      p.intern(normalizeText(get(colDpto), true)),
		Prov:      p.intern(normalizeText(get(colProv), true)),
		Dist:      p.intern(normalizeText(get(colDist), true)),
		Modalidad: p.intern(normalizeText(get(colModalidad), false)),
	}
	if rec.Dpto == "" || rec.Prov == "" || rec.Dist == "" || rec.Modalidad == "" {
		return Record{}, false
	}
	return rec, true
}

func (p *parser) intern(b []byte) string {
	if s, ok := p.strings[string(b)]; ok {
		return s
	}
	s := string(b)
	p.strings[s] = s
	return s
}

func trimEOL(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}

// splitFields separa una línea CSV respetando comillas dobles (con "" como escape).
func splitFields(line []byte, dst [][]byte) [][]byte {
	dst = dst[:0]
	for i := 0; i <= len(line); {
		if i < len(line) && line[i] == '"' {
			j := i + 1
			start := j
			var buf []byte
			for j < len(line) {
				if line[j] == '"' {
					if j+1 < len(line) && line[j+1] == '"' {
						buf = append(buf, line[start:j+1]...)
						j += 2
						start = j
						continue
					}
					break
				}
				j++
			}
			field := line[start:min(j, len(line))]
			if buf != nil {
				field = append(buf, field...)
			}
			dst = append(dst, field)
			for j < len(line) && line[j] != ',' {
				j++
			}
			i = j + 1
			continue
		}
		j := bytes.IndexByte(line[i:], ',')
		if j < 0 {
			dst = append(dst, line[i:])
			break
		}
		dst = append(dst, line[i:i+j])
		i += j + 1
	}
	return dst
}

func isSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\r' || c == '\n' }

// normalizeText recorta espacios, colapsa espacios internos y opcionalmente
// convierte a mayúsculas.
func normalizeText(b []byte, upper bool) []byte {
	b = bytes.TrimFunc(b, func(r rune) bool { return r < 128 && isSpace(byte(r)) })
	needsCollapse := false
	for i := 1; i < len(b); i++ {
		if isSpace(b[i]) && (isSpace(b[i-1]) || b[i] != ' ') {
			needsCollapse = true
			break
		}
	}
	if needsCollapse {
		out := make([]byte, 0, len(b))
		prevSpace := false
		for _, c := range b {
			if isSpace(c) {
				if !prevSpace {
					out = append(out, ' ')
				}
				prevSpace = true
				continue
			}
			prevSpace = false
			out = append(out, c)
		}
		b = out
	}
	if upper && hasLower(b) {
		b = bytes.ToUpper(b)
	}
	return b
}

func hasLower(b []byte) bool {
	for _, c := range b {
		if (c >= 'a' && c <= 'z') || c >= 0x80 {
			return true
		}
	}
	return false
}

// parseInt acepta enteros como "12" o "12.0".
func parseInt(b []byte) (int, bool) {
	b = bytes.TrimSpace(b)
	if dot := bytes.IndexByte(b, '.'); dot >= 0 {
		for _, c := range b[dot+1:] {
			if c != '0' {
				return 0, false
			}
		}
		b = b[:dot]
	}
	if len(b) == 0 || len(b) > 9 {
		return 0, false
	}
	n := 0
	for _, c := range b {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

// parseUbigeo valida un código de 1 a 6 dígitos (se rellena con ceros a la izquierda).
func parseUbigeo(b []byte) (int, bool) {
	n, ok := parseInt(b)
	if !ok || n <= 0 || n > 999999 {
		return 0, false
	}
	return n, true
}
