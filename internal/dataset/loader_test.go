package dataset

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeCSV(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "data.csv")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCSVCleansAndValidates(t *testing.T) {
	content := "\ufeffANIO,MES,DPTO_HECHO_NEW,PROV_HECHO,DIST_HECHO,UBIGEO_HECHO,P_MODALIDADES,cantidad\r\n" +
		`2018,1,amazonas,BAGUA,  ARAMANGO ,10202,"Violencia contra la mujer e integrantes",3` + "\r\n" +
		"2018,1,AMAZONAS,BAGUA,ARAMANGO,10202,Violencia contra la mujer e integrantes,3\n" +
		"2018,13,AMAZONAS,BAGUA,ARAMANGO,10202,Otros,1\n" +
		"2018,2,AMAZONAS,BAGUA,ARAMANGO,abc,Otros,1\n" +
		"2018,2,AMAZONAS,BAGUA,,10202,Otros,1\n" +
		"2019,2.0,LIMA,LIMA,\"SAN  ISIDRO\",150131,Hurto,7.0\n"

	records, stats, err := LoadCSV(writeCSV(t, content), 4)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Lines != 6 || stats.Valid != 2 || stats.Invalid != 3 || stats.Duplicates != 1 {
		t.Fatalf("stats inesperadas: %+v", stats)
	}
	want := Record{Anio: 2018, Mes: 1, Dpto: "AMAZONAS", Prov: "BAGUA", Dist: "ARAMANGO", Ubigeo: 10202,
		Modalidad: "Violencia contra la mujer e integrantes", Cantidad: 3}
	if records[0] != want {
		t.Fatalf("registro 0 = %+v, se esperaba %+v", records[0], want)
	}
	if records[1].Dist != "SAN ISIDRO" || records[1].Mes != 2 || records[1].Cantidad != 7 {
		t.Fatalf("registro 1 mal normalizado: %+v", records[1])
	}
}

// Con muchos bloques pequeños, cada línea debe procesarse exactamente una
// vez sin importar dónde caigan los cortes.
func TestLoadCSVChunkBoundaries(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("ANIO,MES,DPTO_HECHO_NEW,PROV_HECHO,DIST_HECHO,UBIGEO_HECHO,P_MODALIDADES,cantidad\n")
	const n = 20000
	for i := 0; i < n; i++ {
		fmt.Fprintf(&sb, "%d,%d,DPTO,PROV,DIST%d,%d,Robo,%d\n", 2018+i%8, 1+i%12, i, 10101+i, i%5)
	}
	path := writeCSV(t, sb.String())
	for _, workers := range []int{1, 3, 8, 32} {
		records, stats, err := LoadCSV(path, workers)
		if err != nil {
			t.Fatal(err)
		}
		if len(records) != n || stats.Lines != n || stats.Invalid != 0 {
			t.Fatalf("workers=%d: %d registros, stats %+v", workers, len(records), stats)
		}
		for i, r := range records {
			if r.Ubigeo != 10101+i {
				t.Fatalf("workers=%d: orden alterado en %d: %+v", workers, i, r)
			}
		}
	}
}

func TestLoadCSVMissingColumn(t *testing.T) {
	_, _, err := LoadCSV(writeCSV(t, "ANIO,MES\n2018,1\n"), 2)
	if err == nil {
		t.Fatal("se esperaba error por columnas faltantes")
	}
}
