"""Hardware panels shared by offline reports; no analytical dependencies required."""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go

from llm_bench.models import Experiment


def hardware_panels(experiments: list[Experiment]) -> str:
    power, utilization, efficiency, memory = go.Figure(), go.Figure(), go.Figure(), go.Figure()
    thermal = go.Figure()
    cpu_gpu = go.Figure()
    rows = []
    for experiment in experiments:
        if experiment.settings.get("telemetry_host_role") == "client_host":
            rows.append(
                "<tr><td>Client-host telemetry excluded from inference-host comparisons</td></tr>"
            )
            continue
        for sample in experiment.samples:
            if sample.phase not in {"warm_start", "benchmark"}:
                continue
            summaries = [
                s for s in experiment.telemetry_summaries if s.invocation_id == sample.invocation_id
            ]
            label = (
                f"{experiment.model} · {sample.workload} · "
                f"ctx={experiment.settings.get('num_ctx')} · {experiment.experiment_id[:8]}"
            )
            cpus = [s for s in summaries if s.metric == "cpu_utilization" and s.device == "system"]
            for summary in summaries:
                if summary.mean is None:
                    continue
                source_label = f"{label} · {summary.device} · {summary.source}"
                if summary.metric in {"power", "gpu_utilization"}:
                    figure = power if summary.metric == "power" else utilization
                    figure.add_scatter(
                        x=[summary.mean],
                        y=[sample.generation_tokens_per_second],
                        mode="markers",
                        name=source_label,
                    )
                if summary.metric == "gpu_utilization" and cpus:
                    cpu_gpu.add_scatter(
                        x=[cpus[0].mean], y=[summary.mean], mode="markers", name=source_label
                    )
                if summary.metric == "ram_used":
                    memory.add_scatter(
                        x=[experiment.settings.get("num_ctx")],
                        y=[summary.mean],
                        mode="markers",
                        name=source_label,
                    )
                if summary.joules_per_generated_token is not None:
                    efficiency.add_bar(x=[source_label], y=[summary.joules_per_generated_token])
                if summary.metric in {"power", "energy", "gpu_utilization", "temperature"}:
                    rows.append(
                        "<tr>"
                        + "".join(
                            f"<td>{escape(str(value))}</td>"
                            for value in (
                                source_label,
                                sample.repetition,
                                summary.metric,
                                summary.scope,
                                summary.unit,
                                round(summary.mean, 3),
                                round(summary.coverage_fraction, 3),
                                summary.energy_joules,
                                summary.energy_method,
                            )
                        )
                        + "</tr>"
                    )
        # Plot raw thermal/frequency values for the first invocation with these sensors.
        trace_id = next(
            (
                o.invocation_id
                for o in experiment.telemetry
                if o.metric in {"temperature", "gpu_frequency", "cpu_frequency"}
                and o.value is not None
            ),
            None,
        )
        grouped: dict[tuple[str, str, str], tuple[list[float], list[float | None]]] = {}
        for point in experiment.telemetry:
            if point.invocation_id != trace_id or point.metric not in {
                "temperature",
                "gpu_frequency",
                "cpu_frequency",
            }:
                continue
            key = (point.metric, point.device, point.source)
            x, y = grouped.setdefault(key, ([], []))
            x.append(point.offset_seconds)
            y.append(float(point.value) if point.value is not None else None)
        for (metric, device, source), (x, y) in grouped.items():
            thermal.add_scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{experiment.model} · {metric} · {device} · {source}",
                yaxis="y2" if "frequency" in metric else "y",
            )
    panels = []
    for identifier, figure, title, xunit, yunit in (
        (
            "hardware-power",
            power,
            "Throughput versus power (sources kept separate)",
            "W",
            "tokens/s",
        ),
        (
            "hardware-util",
            utilization,
            "Throughput versus GPU utilization",
            "GPU percent",
            "tokens/s",
        ),
        ("hardware-cpu-gpu", cpu_gpu, "CPU versus GPU utilization", "CPU percent", "GPU percent"),
        (
            "hardware-energy",
            efficiency,
            "Whole-invocation energy per generated token",
            "Device / source",
            "J/token",
        ),
        (
            "hardware-memory",
            memory,
            "Memory versus configured context",
            "Context tokens",
            "RAM bytes",
        ),
        (
            "hardware-thermal",
            thermal,
            "Temperature and frequency: selected invocation",
            "Elapsed seconds",
            "degC",
        ),
    ):
        figure.update_layout(title=title, xaxis_title=xunit, yaxis_title=yunit)
        if identifier == "hardware-thermal":
            figure.update_layout(yaxis2={"title": "Hz", "overlaying": "y", "side": "right"})
        if figure.data:
            panels.append(
                figure.to_html(full_html=False, include_plotlyjs=False, div_id=identifier)
            )
    if not panels:
        panels.append(
            "<p>Hardware comparisons unavailable: no supported inference-host measurements.</p>"
        )
    return (
        "<section><h2>Hardware utilization and efficiency</h2>"
        "<p>Sources and overlapping power domains are never summed. "
        "Missing energy remains unavailable. "
        "Energy includes prompt processing and loading. Boundary readings are retained. "
        "Comparison labels identify configuration and experiment; "
        "inspect raw repetitions before pooling.</p>"
        + "".join(panels)
        + "<table><thead><tr><th>Experiment/device/source</th><th>Repetition</th><th>Metric</th>"
        "<th>Scope</th><th>Unit</th><th>Mean</th><th>Coverage</th><th>Energy J</th><th>Method</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )
