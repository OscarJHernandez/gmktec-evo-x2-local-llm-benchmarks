"""Offline, self-contained HTML report generation."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from llm_bench.hardware_report import hardware_panels
from llm_bench.models import Experiment, Summary


def generate_report(experiments: list[Experiment], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    rates: list[float | None] = []
    ttft_values: list[float | None] = []
    durations: list[float | None] = []
    prompt_rates: list[float | None] = []
    load_names: list[str] = []
    load_values: list[float | None] = []
    for experiment in experiments:
        workloads = sorted(
            {
                sample.workload
                for sample in experiment.samples
                if sample.phase in {"benchmark", "warm_start"}
            }
        )
        for workload in workloads:
            names.append(f"{experiment.model} · {workload}")
            rates.append(
                _median(
                    experiment,
                    f"workload:{workload}:generation_tokens_per_second",
                )
            )
            ttft_values.append(_median(experiment, f"workload:{workload}:client_ttft_seconds"))
            durations.append(_median(experiment, f"workload:{workload}:client_duration_seconds"))
            prompt_rates.append(
                _median(
                    experiment,
                    f"workload:{workload}:prompt_tokens_per_second_uncached",
                )
            )
        if "cold_start" in {sample.phase for sample in experiment.samples}:
            load_names.append(experiment.model)
            load_values.append(_median(experiment, "cold_start_load_seconds"))
    generation = go.Figure(go.Bar(x=names, y=rates))
    generation.update_layout(title="Generation throughput (median)", yaxis_title="tokens/s")
    ttft = go.Figure(go.Bar(x=names, y=ttft_values))
    ttft.update_layout(title="Client-observed time to first output", yaxis_title="seconds")
    prompt = go.Figure(go.Bar(x=names, y=prompt_rates))
    prompt.update_layout(
        title="Prompt throughput — known uncached only", yaxis_title="input tokens/s"
    )
    latency = go.Figure()
    latency.add_bar(
        x=names,
        y=durations,
        name="Client end-to-end",
    )
    latency.add_bar(
        x=load_names,
        y=load_values,
        name="Server cold model load",
    )
    latency.update_layout(
        title="End-to-end duration and model load", yaxis_title="seconds", barmode="group"
    )

    rows: list[str] = []
    for experiment in experiments:
        workloads = sorted(
            {
                sample.workload
                for sample in experiment.samples
                if sample.phase in {"benchmark", "warm_start"}
            }
        )
        for workload in workloads:
            rate = experiment.summaries.get(f"workload:{workload}:generation_tokens_per_second")
            ttft_summary = experiment.summaries.get(f"workload:{workload}:client_ttft_seconds")
            uncached_prompt = experiment.summaries.get(
                f"workload:{workload}:prompt_tokens_per_second_uncached"
            )
            cached_prompt = experiment.summaries.get(
                f"workload:{workload}:prompt_tokens_per_second_cached"
            )
            unknown_prompt = experiment.summaries.get(
                f"workload:{workload}:prompt_tokens_per_second_cache_unknown"
            )
            rows.append(
                "<tr>"
                f"<td>{_escape(experiment.model)}</td><td>{_escape(workload)}</td>"
                f"<td>{_escape(experiment.model_digest or 'not reported')}</td>"
                f"<td>{_stat(rate, 'mean')}</td><td>{_stat(rate, 'median')}</td>"
                f"<td>{_stat(rate, 'standard_deviation')}</td>"
                f"<td>{_stat(ttft_summary, 'median')}</td>"
                f"<td>{_stat(uncached_prompt, 'median')}</td>"
                f"<td>{_stat(cached_prompt, 'median')}</td>"
                f"<td>{_stat(unknown_prompt, 'median')}</td></tr>"
            )

    warnings = "".join(
        f"<li>{_escape(text)}</li>" for item in experiments for text in item.warnings
    )
    synthetic = any(item.synthetic for item in experiments)
    warning_banner = (
        '<p class="notice">SYNTHETIC EXAMPLE DATA — values are illustrative, '
        "not measured benchmarks.</p>"
        if synthetic
        else ""
    )
    configurations = []
    for experiment in experiments:
        ram_used = (
            str(experiment.hardware.ram_percent)
            if experiment.hardware.ram_percent is not None
            else "unknown"
        )
        gpu_bytes = (
            str(experiment.hardware.gpu_memory_bytes)
            if experiment.hardware.gpu_memory_bytes is not None
            else "not reported"
        )
        configurations.append(
            f"<section><h3>{_escape(experiment.model)}</h3>"
            f"<p>Ollama {_escape(experiment.backend_version or 'unknown')} · "
            f"{experiment.repetition_count} repetitions · {experiment.warmup_count} warmups</p>"
            f"<p>OS: {_escape(experiment.hardware.platform)} · "
            f"CPU: {_escape(experiment.hardware.cpu)} · "
            f"RAM: {experiment.hardware.total_ram_bytes or 0:,} bytes total, "
            f"{experiment.hardware.available_ram_bytes or 0:,} available, "
            f"{_escape(ram_used)}% used. "
            f"GPU: {_escape(experiment.hardware.gpu or 'not reported')}; "
            f"GPU memory: {_escape(gpu_bytes)} bytes "
            f"({_escape(experiment.hardware.gpu_memory_source or 'no collector')}).</p>"
            f"<pre>{_escape(str(experiment.settings))}</pre></section>"
        )
    output.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Local LLM benchmark report</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#192431;background:#f3f6fa;margin:0}}
main{{max-width:1100px;margin:auto;padding:2rem}}h1{{font-size:2.25rem}}section,.panel{{background:white;
border-radius:12px;padding:1.4rem;margin:1rem 0;box-shadow:0 2px 12px #18334a12}}
table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:.7rem;border-bottom:1px solid #e4eaf0}}
.notice{{background:#fff0ce;padding:1rem;font-weight:700}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
@media(max-width:700px){{.charts{{grid-template-columns:1fr}}}}
</style><script>{get_plotlyjs()}</script></head><body><main>
<h1>Local LLM benchmark report</h1>
<p>Client and server timings have distinct sources; raw samples are retained below.</p>
{warning_banner}<section><h2>Executive overview</h2><div class="charts">
<div id="generation"></div><div id="prompt"></div><div id="ttft"></div><div id="duration"></div>
</div></section>
<section><h2>Model comparison by workload</h2><table><thead><tr>
<th>Model</th><th>Workload</th><th>Digest</th><th>Mean output tok/s</th>
<th>Median output tok/s</th><th>Std dev tok/s</th><th>Median client TTFT (s)</th>
<th>Uncached input tok/s</th><th>Cached input tok/s</th><th>Unknown-cache input tok/s</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></section>
{hardware_panels(experiments)}
<section><h2>Experiment and hardware settings</h2>{"".join(configurations)}</section>
<section><h2>Variability and individual runs</h2>{_raw_table(experiments)}</section>
<section><h2>Warnings and limitations</h2><ul>{warnings}</ul>
<p>Prompt cache telemetry is not uniformly exposed by Ollama. GPU memory is shown
only when a collector reports actual consumption; address-space limits are not
physical usage.</p></section>
</main><script>
Plotly.newPlot('generation',{_safe_json(generation.to_json())},{{responsive:true}});
Plotly.newPlot('prompt',{_safe_json(prompt.to_json())},{{responsive:true}});
Plotly.newPlot('ttft',{_safe_json(ttft.to_json())},{{responsive:true}});
Plotly.newPlot('duration',{_safe_json(latency.to_json())},{{responsive:true}});
</script></body></html>""",
        encoding="utf-8",
    )
    return output


def _median(experiment: Experiment, key: str) -> float | None:
    summary = experiment.summaries.get(key)
    return summary.median if summary is not None else None


def _stat(summary: Summary | None, name: str) -> str:
    value = getattr(summary, name) if summary is not None else None
    return f"{value:.3f}" if value is not None else "n/a"


def _safe_json(value: str) -> str:
    return value.replace("</", "<\\/")


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _raw_table(experiments: list[Experiment]) -> str:
    rows = []
    for experiment in experiments:
        for sample in experiment.samples:
            if sample.phase not in {"benchmark", "warm_start", "cold_start"}:
                continue
            rows.append(
                "<tr>"
                + "".join(
                    f"<td>{_escape(str(value))}</td>"
                    for value in (
                        experiment.model,
                        sample.phase,
                        sample.workload,
                        sample.repetition,
                        sample.generation_tokens_per_second,
                        sample.prompt_tokens_per_second,
                        sample.client_ttft_seconds,
                        sample.client_duration_seconds,
                        sample.server_load_seconds,
                        sample.server_total_seconds,
                        sample.prompt_cache_hit,
                        sample.premature_completion,
                    )
                )
                + "</tr>"
            )
    headings = (
        "Model",
        "Phase",
        "Workload",
        "Run",
        "Output tok/s",
        "Input tok/s",
        "Client TTFT s",
        "Client duration s",
        "Server load s",
        "Server total s",
        "Cache hit",
        "Premature",
    )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{heading}</th>" for heading in headings)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
