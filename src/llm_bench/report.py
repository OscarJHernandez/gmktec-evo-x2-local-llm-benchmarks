"""Offline, self-contained HTML report generation."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from llm_bench.models import Experiment


def generate_report(experiments: list[Experiment], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [item.model for item in experiments]
    rates = [
        item.summaries.get("generation_tokens_per_second").median
        if item.summaries.get("generation_tokens_per_second")
        else None
        for item in experiments
    ]
    latency = [
        item.summaries.get("client_ttft_seconds").median
        if item.summaries.get("client_ttft_seconds")
        else None
        for item in experiments
    ]
    generation = go.Figure(go.Bar(x=names, y=rates, name="Output tokens/s"))
    generation.update_layout(title="Generation throughput (median)", yaxis_title="tokens/s")
    ttft = go.Figure(go.Bar(x=names, y=latency, name="Client-observed TTFT"))
    ttft.update_layout(title="Time to first output (median)", yaxis_title="seconds")
    tables = []
    for experiment in experiments:
        rate = experiment.summaries.get("generation_tokens_per_second")
        ttft_summary = experiment.summaries.get("client_ttft_seconds")
        tables.append(
            "<tr>"
            f"<td>{_escape(experiment.model)}</td>"
            f"<td>{_escape(experiment.model_digest or 'not reported')}</td>"
            f"<td>{rate.median:.2f}</td>" if rate and rate.median is not None else
            f"<td>{_escape(experiment.model)}</td><td>{_escape(experiment.model_digest or 'not reported')}</td><td>n/a</td>"
        )
        tables[-1] += f"<td>{ttft_summary.median:.3f}</td>" if ttft_summary and ttft_summary.median is not None else "<td>n/a</td>"
        tables[-1] += "</tr>"
    warnings = "".join(f"<li>{_escape(text)}</li>" for item in experiments for text in item.warnings)
    synthetic = any(item.synthetic for item in experiments)
    warning_banner = (
        '<p class="notice">SYNTHETIC EXAMPLE DATA — values are illustrative, not measured benchmarks.</p>'
        if synthetic else ""
    )
    configurations = "".join(
        f"<section><h3>{_escape(item.model)}</h3><p>Ollama { _escape(item.backend_version or 'unknown')}"
        f" · {item.repetition_count} repetitions · {item.warmup_count} warmups"
        f" · {item.hardware.total_ram_bytes or 0:,} bytes RAM</p>"
        f"<pre>{_escape(str(item.settings))}</pre></section>"
        for item in experiments
    )
    output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Local LLM benchmark report</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#192431;background:#f3f6fa;margin:0}}
main{{max-width:1100px;margin:auto;padding:2rem}}h1{{font-size:2.25rem}}section,.panel{{background:white;
border-radius:12px;padding:1.4rem;margin:1rem 0;box-shadow:0 2px 12px #18334a12}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:.7rem;border-bottom:1px solid #e4eaf0}}
.notice{{background:#fff0ce;padding:1rem;font-weight:700}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}@media(max-width:700px){{.charts{{grid-template-columns:1fr}}}}
</style><script>{get_plotlyjs()}</script></head><body><main>
<h1>Local LLM benchmark report</h1><p>Measurements collected by llm-bench. Client and server timing sources are shown separately.</p>
{warning_banner}<section><h2>Executive overview</h2><div class="charts"><div id="generation"></div><div id="ttft"></div></div></section>
<section><h2>Model comparison</h2><table><thead><tr><th>Model</th><th>Digest</th><th>Median output tokens/s</th>
<th>Median client TTFT (s)</th></tr></thead><tbody>{''.join(tables)}</tbody></table></section>
<section><h2>Experiment and hardware settings</h2>{configurations}</section>
<section><h2>Variability and individual runs</h2>{_raw_table(experiments)}</section>
<section><h2>Warnings and limitations</h2><ul>{warnings}</ul><p>Prompt cache telemetry is not uniformly exposed by Ollama. GPU memory is omitted unless a collector supplies actual consumption; reported memory limits are not physical usage.</p></section>
</main><script>
Plotly.newPlot('generation',{generation.to_json()},{{responsive:true}});
Plotly.newPlot('ttft',{ttft.to_json()},{{responsive:true}});
</script></body></html>""",
        encoding="utf-8",
    )
    return output


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#x27;")


def _raw_table(experiments: list[Experiment]) -> str:
    rows = []
    for experiment in experiments:
        for sample in experiment.samples:
            if sample.phase != "benchmark":
                continue
            rows.append(
                "<tr>" + "".join(
                    f"<td>{_escape(str(value))}</td>"
                    for value in (
                        experiment.model, sample.workload, sample.repetition,
                        sample.generation_tokens_per_second, sample.client_ttft_seconds,
                        sample.client_duration_seconds, sample.prompt_cache_hit,
                        sample.premature_completion,
                    )
                ) + "</tr>"
            )
    headings = ("Model", "Workload", "Run", "Output tok/s", "Client TTFT s",
                "Client duration s", "Cache hit", "Premature")
    return "<table><thead><tr>" + "".join(f"<th>{heading}</th>" for heading in headings) + \
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
