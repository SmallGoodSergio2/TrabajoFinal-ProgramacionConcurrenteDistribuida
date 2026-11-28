package docs

import _ "embed"

//go:embed openapi.yaml
var OpenAPI []byte

//go:embed swagger.html
var SwaggerHTML string
