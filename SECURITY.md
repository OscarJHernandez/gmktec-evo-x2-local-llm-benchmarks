# Security policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public issues. Use GitHub's
private vulnerability reporting feature for this repository. Include steps to
reproduce and impact where possible. Do not include credentials or private
machine data in a report.

## Security design

The CLI communicates only with the explicitly configured Ollama API endpoint. It
does not download models, execute shell commands from benchmark settings, or
expose an Ollama listener. Result JSON intentionally excludes the API URL.
Review prompts and results before sharing them; future user-supplied prompts
could contain private text.
