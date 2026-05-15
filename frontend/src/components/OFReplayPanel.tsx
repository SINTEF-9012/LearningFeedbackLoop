import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type Plotly from "plotly.js";
import type { PipelineConfig } from "../types";
import {
  getMachines,
  getOFs,
  getOFWindows,
  type MachineInfo,
  type OFWindow,
} from "../api";
import {
  Card,
  Field,
  inputClass,
  btnPrimary,
  btnDanger,
  btnSecondary,
  Badge,
  usePlotly,
} from "../ui";

interface Props {
  config: PipelineConfig;
}

interface OFStep {
  type: "step";
  t: number;
  ts: string;
  pairs: number[][][]; // (C, K, 2)
  prob: number | null;
  inference_ms: number | null;
  tool_number: number | null;
  tool_description: string | null;
  diameter_mm: number | null;
  n_inserts: number | null;
  spindle_rpm: number | null;
  feed_rate: number | null;
  operation_mode: number | null;
  valid: boolean;
  tool_changed: boolean;
  params: Record<string, number>;
}

interface OFInit {
  type: "init";
  total_steps: number;
  n_channels: number;
  k_peaks: number;
  channel_names: string[];
  cnn_window: number;
  machine_id: string;
  of: string;
  start: string;
  end: string;
  f_max_rel: number | null;
}

const CHANNEL_COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

export default function OFReplayPanel({ config }: Props) {
  const [machines, setMachines] = useState<MachineInfo[]>([]);
  const [machineId, setMachineId] = useState("");
  const [ofs, setOfs] = useState<string[]>([]);
  const [of, setOf] = useState("");
  const [windows, setWindows] = useState<OFWindow[]>([]);
  const [winIdx, setWinIdx] = useState<number>(-1);
  const [loadingWindows, setLoadingWindows] = useState(false);
  const [windowErr, setWindowErr] = useState<string | null>(null);

  const [speed, setSpeed] = useState(50);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [initData, setInitData] = useState<OFInit | null>(null);

  const [timeSteps, setTimeSteps] = useState<number[]>([]);
  const [tsLabels, setTsLabels] = useState<string[]>([]);
  const [probVals, setProbVals] = useState<(number | null)[]>([]);
  const [inferenceTimes, setInferenceTimes] = useState<number[]>([]);
  const [pairsHistory, setPairsHistory] = useState<number[][][][]>([]);
  const [spindleHistory, setSpindleHistory] = useState<(number | null)[]>([]);
  const [toolChanges, setToolChanges] = useState<
    { step: number; from: number | null; to: number | null }[]
  >([]);
  const [lastStep, setLastStep] = useState<OFStep | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // -- Load machines on mount -------------------------------------------------
  useEffect(() => {
    getMachines()
      .then((d) => {
        setMachines(d.machines || []);
        const first = (d.machines || []).find((m) => m.available);
        if (first) setMachineId(first.id);
      })
      .catch(() => {});
  }, []);

  // -- Load OFs when machine changes -----------------------------------------
  useEffect(() => {
    if (!machineId) return;
    getOFs(machineId).then((d) => {
      setOfs(d.ofs || []);
      setOf(d.ofs?.[0] ?? "");
      setWindows([]);
      setWinIdx(-1);
    });
  }, [machineId]);

  const fetchWindows = useCallback(async () => {
    if (!machineId || !of) return;
    setLoadingWindows(true);
    setWindowErr(null);
    try {
      const d = await getOFWindows({
        machine_id: machineId,
        of,
      });
      if (d.error) {
        setWindowErr(d.error);
        setWindows([]);
      } else {
        setWindows(d.windows);
        setWinIdx(d.windows.length > 0 ? 0 : -1);
      }
    } finally {
      setLoadingWindows(false);
    }
  }, [machineId, of]);

  // Auto-fetch when OF changes (but not on every active-mode keystroke).
  useEffect(() => {
    if (machineId && of) fetchWindows();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [machineId, of]);

  const reset = useCallback(() => {
    setTimeSteps([]);
    setTsLabels([]);
    setProbVals([]);
    setPairsHistory([]);
    setInferenceTimes([]);
    setSpindleHistory([]);
    setToolChanges([]);
    setInitData(null);
    setLastStep(null);
    setRunning(false);
    setPaused(false);
  }, []);

  const handleStart = useCallback(() => {
    if (winIdx < 0 || !windows[winIdx]) return;
    reset();
    const w = windows[winIdx];
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/of_replay`);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          action: "start",
          machine_id: machineId,
          of,
          start: w.start,
          end: w.end,
          speed,
        })
      );
      setRunning(true);
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "init") {
        setInitData(msg as OFInit);
      } else if (msg.type === "step") {
        const step = msg as OFStep;
        setTimeSteps((p) => [...p, step.t]);
        setTsLabels((p) => [...p, step.ts]);
        setProbVals((p) => [...p, step.prob]);
        setPairsHistory((p) => [...p, step.pairs]);
        setSpindleHistory((p) => [...p, step.spindle_rpm]);
        if (step.inference_ms !== null) {
          setInferenceTimes((p) => [...p, step.inference_ms!]);
        }
        if (step.tool_changed) {
          setToolChanges((p) => [
            ...p,
            {
              step: step.t,
              from: p.length > 0 ? p[p.length - 1].to : null,
              to: step.tool_number,
            },
          ]);
        }
        setLastStep(step);
      } else if (msg.type === "done") {
        setRunning(false);
        setPaused(false);
      } else if (msg.type === "error") {
        alert(msg.message);
        setRunning(false);
      }
    };

    ws.onclose = () => {
      setRunning(false);
      setPaused(false);
    };
  }, [winIdx, windows, machineId, of, speed, reset]);

  const handlePause = () => {
    wsRef.current?.send(JSON.stringify({ action: "pause" }));
    setPaused(true);
  };
  const handleResume = () => {
    wsRef.current?.send(JSON.stringify({ action: "resume" }));
    setPaused(false);
  };
  const handleStop = () => {
    wsRef.current?.send(JSON.stringify({ action: "stop" }));
    wsRef.current?.close();
    setRunning(false);
    setPaused(false);
  };
  const handleSpeedChange = (n: number) => {
    setSpeed(n);
    if (running) wsRef.current?.send(JSON.stringify({ action: "set_speed", speed: n }));
  };

  useEffect(() => () => wsRef.current?.close(), []);

  const validProbs = probVals.filter((p): p is number => p !== null);
  const validProbSteps = timeSteps.filter((_, i) => probVals[i] !== null);
  const latestProb = validProbs.length > 0 ? validProbs[validProbs.length - 1] : null;
  const progress = initData
    ? Math.round((timeSteps.length / initData.total_steps) * 100)
    : 0;
  const avgInferenceMs =
    inferenceTimes.length > 0
      ? inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length
      : null;

  // ---- Probability chart with tool-change markers -------------------------
  const probData: Plotly.Data[] = useMemo(() => {
    const traces: Plotly.Data[] = [
      {
        x: validProbSteps,
        y: validProbs,
        type: "scatter",
        mode: "lines",
        name: "P(broke)",
        line: { color: "#111827", width: 2 },
      },
      {
        x: initData ? [0, initData.total_steps] : [],
        y: [0.5, 0.5],
        type: "scatter",
        mode: "lines",
        name: "threshold",
        line: { color: "#9ca3af", width: 1, dash: "dash" },
        showlegend: false,
      },
    ];
    if (toolChanges.length > 0) {
      // Plot as vertical stems via a single trace with None separators.
      const xs: (number | null)[] = [];
      const ys: (number | null)[] = [];
      const texts: string[] = [];
      for (const tc of toolChanges) {
        xs.push(tc.step, tc.step, null);
        ys.push(0, 1, null);
        texts.push(`tool ${tc.from ?? "?"} → ${tc.to ?? "?"}`, "", "");
      }
      traces.push({
        x: xs,
        y: ys,
        type: "scatter",
        mode: "lines",
        line: { color: "#ef4444", width: 1, dash: "dot" },
        name: "tool change",
        hovertext: texts,
        hoverinfo: "text",
        showlegend: true,
      });
    }
    return traces;
  }, [validProbSteps, validProbs, initData, toolChanges]);

  const probLayout: Partial<Plotly.Layout> = {
    title: { text: "Break probability over time", font: { size: 13 } },
    xaxis: {
      title: { text: "Step (≈ s)" },
      range: initData ? [0, initData.total_steps] : undefined,
    },
    yaxis: { title: { text: "P(broke)" }, range: [-0.05, 1.05] },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 260,
    legend: { orientation: "h", y: -0.25 },
  };
  const probRef = usePlotly(probData, probLayout);

  // ---- Peak-pair scatter: f_Hz vs time ------------------------------------
  const peakScatterData: Plotly.Data[] = useMemo(() => {
    if (!initData || pairsHistory.length === 0) return [];
    const C = initData.n_channels;
    // Per-channel max amp for marker scaling
    const maxAmps = Array(C).fill(1e-9);
    for (const step of pairsHistory) {
      for (let c = 0; c < C; c++) {
        for (const pk of step[c]) {
          if (pk[1] > maxAmps[c]) maxAmps[c] = pk[1];
        }
      }
    }
    const traces: Plotly.Data[] = [];
    for (let c = 0; c < C; c++) {
      const xs: number[] = [];
      const ys: number[] = [];
      const sizes: number[] = [];
      const customs: number[] = [];
      for (let t = 0; t < pairsHistory.length; t++) {
        const step = pairsHistory[t];
        const tNum = timeSteps[t] ?? t;
        const fg = (spindleHistory[t] ?? 0) / 60;
        if (fg <= 0) continue;
        for (const pk of step[c]) {
          if (pk[1] <= 0) continue;
          xs.push(tNum);
          ys.push(pk[0] * fg);
          sizes.push(4 + 18 * (pk[1] / maxAmps[c]));
          customs.push(pk[1]);
        }
      }
      traces.push({
        x: xs,
        y: ys,
        type: "scatter",
        mode: "markers",
        name: initData.channel_names[c] ?? `ch${c}`,
        marker: {
          color: CHANNEL_COLORS[c % CHANNEL_COLORS.length],
          size: sizes,
          opacity: 0.55,
          line: { width: 0 },
        },
        customdata: customs,
        hovertemplate:
          "t=%{x}<br>f=%{y:.1f} Hz<br>amp=%{customdata:.2f}<extra>%{fullData.name}</extra>",
      });
    }
    return traces;
  }, [initData, pairsHistory, timeSteps, spindleHistory]);

  const peakLayout: Partial<Plotly.Layout> = {
    title: {
      text: "Peak-pair stream — frequency (Hz) vs step, marker size ∝ amplitude",
      font: { size: 13 },
    },
    xaxis: {
      title: { text: "Step (≈ s)" },
      range: initData ? [0, initData.total_steps] : undefined,
    },
    yaxis: { title: { text: "Frequency (Hz)" } },
    margin: { t: 40, r: 20, b: 60, l: 60 },
    height: 340,
    legend: { orientation: "h", y: -0.2 },
  };
  const peakRef = usePlotly(peakScatterData, peakLayout);

  // ---- Latest snapshot (stem plot in Hz) ----------------------------------
  const latestPairs = pairsHistory[pairsHistory.length - 1];
  const latestSpindle = spindleHistory[spindleHistory.length - 1];
  const snapshotData: Plotly.Data[] = useMemo(() => {
    if (!initData || !latestPairs || !latestSpindle) return [];
    const fg = latestSpindle / 60;
    const traces: Plotly.Data[] = [];
    for (let c = 0; c < initData.n_channels; c++) {
      const xs: (number | null)[] = [];
      const ys: (number | null)[] = [];
      for (const pk of latestPairs[c]) {
        if (pk[1] <= 0) continue;
        const fHz = pk[0] * fg;
        xs.push(fHz, fHz, null);
        ys.push(0, pk[1], null);
      }
      traces.push({
        x: xs,
        y: ys,
        type: "scatter",
        mode: "lines+markers",
        name: initData.channel_names[c] ?? `ch${c}`,
        line: { color: CHANNEL_COLORS[c % CHANNEL_COLORS.length], width: 2 },
        marker: { size: 6, color: CHANNEL_COLORS[c % CHANNEL_COLORS.length] },
        connectgaps: false,
      });
    }
    return traces;
  }, [initData, latestPairs, latestSpindle]);

  const snapshotLayout: Partial<Plotly.Layout> = {
    title: {
      text: lastStep
        ? `Current spectrum @ t=${lastStep.t} — top-${initData?.k_peaks ?? 0} peaks per channel`
        : "Current spectrum",
      font: { size: 13 },
    },
    xaxis: { title: { text: "Frequency (Hz)" } },
    yaxis: { title: { text: "Amplitude" } },
    margin: { t: 40, r: 20, b: 60, l: 60 },
    height: 280,
    legend: { orientation: "h", y: -0.25 },
  };
  const snapshotRef = usePlotly(snapshotData, snapshotLayout);

  const selWin = winIdx >= 0 ? windows[winIdx] : null;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">OF replay (real-machine streaming inference)</h2>

      <Card>
        <div className="grid grid-cols-12 gap-3 items-end">
          <div className="col-span-3">
            <Field label="Machine">
              <select
                className={inputClass}
                value={machineId}
                onChange={(e) => setMachineId(e.target.value)}
              >
                {machines.map((m) => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name} {m.available ? `(${m.n_ofs} OF)` : "(unavailable)"}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="col-span-2">
            <Field label="Fabrication order">
              <select
                className={inputClass}
                value={of}
                onChange={(e) => setOf(e.target.value)}
              >
                {ofs.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="col-span-2 flex gap-2">
            <button className={btnSecondary} onClick={fetchWindows} disabled={loadingWindows}>
              {loadingWindows ? "Loading…" : "Refresh windows"}
            </button>
          </div>
        </div>
        {windowErr && (
          <div className="mt-3 text-xs text-red-600">{windowErr}</div>
        )}
      </Card>

      <Card title={`Cutting windows (${windows.length})`}>
        {windows.length === 0 ? (
          <div className="text-xs text-gray-500">
            No active-cutting windows found for the current filters.
          </div>
        ) : (
          <div className="max-h-64 overflow-auto border border-gray-100 rounded">
            <table className="w-full text-xs">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-2 py-1 text-left">#</th>
                  <th className="px-2 py-1 text-left">Start (UTC)</th>
                  <th className="px-2 py-1 text-left">End (UTC)</th>
                  <th className="px-2 py-1 text-right">Duration</th>
                  <th className="px-2 py-1"></th>
                </tr>
              </thead>
              <tbody>
                {windows.map((w, i) => (
                  <tr
                    key={i}
                    className={`border-t border-gray-100 cursor-pointer ${
                      winIdx === i ? "bg-indigo-50" : ""
                    }`}
                    onClick={() => setWinIdx(i)}
                  >
                    <td className="px-2 py-1 text-gray-500">{i + 1}</td>
                    <td className="px-2 py-1 font-mono">{w.start.slice(0, 19).replace("T", " ")}</td>
                    <td className="px-2 py-1 font-mono">{w.end.slice(0, 19).replace("T", " ")}</td>
                    <td className="px-2 py-1 text-right">{fmtDuration(w.duration_sec)}</td>
                    <td className="px-2 py-1">
                      {winIdx === i && <Badge color="blue">selected</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card>
        <div className="flex items-center gap-3 flex-wrap">
          <button
            className={btnPrimary}
            onClick={handleStart}
            disabled={running || winIdx < 0}
          >
            Start replay
          </button>
          {running && !paused && (
            <button className={btnSecondary} onClick={handlePause}>
              Pause
            </button>
          )}
          {running && paused && (
            <button className={btnSecondary} onClick={handleResume}>
              Resume
            </button>
          )}
          {running && (
            <button className={btnDanger} onClick={handleStop}>
              Stop
            </button>
          )}
          <Field label={`Speed (steps/s) — ${speed}`}>
            <input
              type="range"
              min={1}
              max={500}
              value={speed}
              onChange={(e) => handleSpeedChange(Number(e.target.value))}
              className="w-48"
            />
          </Field>
          {running && (
            <div className="text-xs text-gray-600">
              Step {timeSteps.length} / {initData?.total_steps ?? "?"} ({progress}%)
            </div>
          )}
          {avgInferenceMs !== null && (
            <Badge color="gray">avg inference {avgInferenceMs.toFixed(2)} ms</Badge>
          )}
          {selWin && (
            <Badge color="gray">
              window {winIdx + 1}: {fmtDuration(selWin.duration_sec)}
            </Badge>
          )}
        </div>
      </Card>

      {/* Live current-step readout */}
      {lastStep && (
        <Card title="Current step">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <Stat label="Timestamp" value={lastStep.ts.slice(0, 19).replace("T", " ")} />
            <Stat
              label="P(broke)"
              value={
                latestProb !== null
                  ? latestProb.toFixed(3)
                  : "warming up…"
              }
              color={
                latestProb !== null && latestProb > 0.5 ? "red" : "gray"
              }
            />
            <Stat
              label="Tool"
              value={
                lastStep.tool_number !== null
                  ? `T${lastStep.tool_number}${
                      lastStep.tool_description
                        ? ` — ${lastStep.tool_description}`
                        : ""
                    }`
                  : "—"
              }
            />
            <Stat
              label="Diameter / teeth"
              value={
                lastStep.diameter_mm
                  ? `Ø${lastStep.diameter_mm} mm / z=${lastStep.n_inserts ?? "?"}`
                  : "—"
              }
            />
            <Stat
              label="Spindle"
              value={lastStep.spindle_rpm ? `${lastStep.spindle_rpm.toFixed(0)} rpm` : "—"}
            />
            <Stat
              label="Feed"
              value={
                lastStep.feed_rate
                  ? `${lastStep.feed_rate.toFixed(1)} mm/min`
                  : "—"
              }
            />
            <Stat
              label="f (per tooth)"
              value={
                lastStep.params.f
                  ? `${lastStep.params.f.toFixed(4)} mm`
                  : "—"
              }
            />
            <Stat
              label="Operation mode"
              value={
                lastStep.operation_mode !== null
                  ? lastStep.operation_mode.toString()
                  : "—"
              }
            />
          </div>
        </Card>
      )}

      {/* Plots */}
      <Card>
        <div ref={probRef} />
      </Card>
      <Card>
        <div ref={peakRef} />
      </Card>
      <Card>
        <div ref={snapshotRef} />
      </Card>
    </div>
  );
}

function fmtDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec - m * 60);
    return `${m}m ${s}s`;
  }
  const h = Math.floor(sec / 3600);
  const m = Math.round((sec - h * 3600) / 60);
  return `${h}h ${m}m`;
}

function Stat({
  label,
  value,
  color = "gray",
}: {
  label: string;
  value: string;
  color?: "gray" | "red";
}) {
  return (
    <div className="bg-gray-50 rounded p-2 border border-gray-100">
      <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      <div
        className={`mt-0.5 font-mono text-sm ${
          color === "red" ? "text-red-600" : "text-gray-900"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
