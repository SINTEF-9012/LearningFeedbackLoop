import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type Plotly from "plotly.js";
import type { PipelineConfig, FolderInfo, FileInfo, SimInit, SimStep } from "../types";
import { getFolders, getFiles, getTestFiles } from "../api";
import type { TestFileInfo } from "../api";
import { Card, Field, inputClass, btnPrimary, btnDanger, btnSecondary, Badge, usePlotly } from "../ui";

type FileSource = "folders" | "test_set";

interface Props {
  config: PipelineConfig;
}

const CHANNEL_COLORS = ["#3b82f6", "#10b981", "#f59e0b"];

export default function SimulationPanel({ config }: Props) {
  const [fileSource, setFileSource] = useState<FileSource>("folders");
  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [selFolder, setSelFolder] = useState("");
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selFile, setSelFile] = useState("");
  const [testFiles, setTestFiles] = useState<TestFileInfo[]>([]);
  const [selTestFile, setSelTestFile] = useState("");
  const [speed, setSpeed] = useState(10);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [initData, setInitData] = useState<SimInit | null>(null);

  const [timeSteps, setTimeSteps] = useState<number[]>([]);
  const [probVals, setProbVals] = useState<(number | null)[]>([]);
  const [inferenceTimes, setInferenceTimes] = useState<number[]>([]);
  // pairsHistory[t] = SimStep["pairs"]  shape (C, K, 2)
  const [pairsHistory, setPairsHistory] = useState<number[][][][]>([]);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    getFolders().then((d) => setFolders(d.folders || [])).catch(() => {});
    getTestFiles().then((d) => {
      setTestFiles(d.files || []);
      if (d.files?.length) setSelTestFile(d.files[0].path);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selFolder) return;
    getFiles(selFolder).then((d) => {
      setFiles(d.files || []);
      if (d.files?.length) setSelFile(d.files[0].name);
    });
  }, [selFolder]);

  useEffect(() => {
    if (folders.length && !selFolder) setSelFolder(folders[0].name);
  }, [folders, selFolder]);

  const reset = useCallback(() => {
    setTimeSteps([]);
    setProbVals([]);
    setPairsHistory([]);
    setInferenceTimes([]);
    setInitData(null);
    setRunning(false);
    setPaused(false);
  }, []);

  const handleStart = useCallback(() => {
    reset();
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/simulate`);
    wsRef.current = ws;

    const filePath = fileSource === "test_set" ? selTestFile : `${selFolder}/${selFile}`;

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "start", file_path: filePath, speed }));
      setRunning(true);
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "init") {
        setInitData(msg as SimInit);
      } else if (msg.type === "step") {
        const step = msg as SimStep;
        setTimeSteps((p) => [...p, step.t]);
        setProbVals((p) => [...p, step.prob]);
        setPairsHistory((p) => [...p, step.pairs]);
        if (step.inference_ms !== null) {
          setInferenceTimes((p) => [...p, step.inference_ms!]);
        }
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
  }, [fileSource, selFolder, selFile, selTestFile, speed, reset]);

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
  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    if (running) wsRef.current?.send(JSON.stringify({ action: "set_speed", speed: newSpeed }));
  };

  useEffect(() => () => wsRef.current?.close(), []);

  const validProbs = probVals.filter((p): p is number => p !== null);
  const validProbSteps = timeSteps.filter((_, i) => probVals[i] !== null);
  const latestProb = validProbs.length > 0 ? validProbs[validProbs.length - 1] : null;
  const progress = initData ? Math.round((timeSteps.length / initData.total_steps) * 100) : 0;
  const avgInferenceMs = inferenceTimes.length > 0
    ? inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length
    : null;

  // -- Probability chart --
  const probData: Plotly.Data[] = [
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
  const probLayout: Partial<Plotly.Layout> = {
    title: { text: "Break probability over time", font: { size: 13 } },
    xaxis: { title: { text: "Time step" }, range: initData ? [0, initData.total_steps] : undefined },
    yaxis: { title: { text: "P(broke)" }, range: [-0.05, 1.05] },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 260,
  };
  const probRef = usePlotly(probData, probLayout);

  // -- Peak-pair scatter per channel: x=t, y=f_Hz, marker size=amp --
  // We build this once per pairsHistory change. Keep the markers reasonable
  // by scaling amplitudes globally per channel (largest amp -> ~20px).
  const peakScatterData: Plotly.Data[] = useMemo(() => {
    if (!initData || pairsHistory.length === 0) return [];
    const fg = initData.spindle_freq;
    const C = initData.n_channels;
    const traces: Plotly.Data[] = [];
    // First pass: per-channel max amplitude for marker scaling.
    const maxAmps = Array(C).fill(1e-9);
    for (const tStep of pairsHistory) {
      for (let c = 0; c < C; c++) {
        for (const pk of tStep[c]) {
          if (pk[1] > maxAmps[c]) maxAmps[c] = pk[1];
        }
      }
    }
    for (let c = 0; c < C; c++) {
      const xs: number[] = [];
      const ys: number[] = [];
      const sizes: number[] = [];
      const customs: number[] = [];
      for (let t = 0; t < pairsHistory.length; t++) {
        const tStep = pairsHistory[t];
        const tNum = timeSteps[t] ?? t;
        for (const pk of tStep[c]) {
          const fRel = pk[0];
          const amp = pk[1];
          if (amp <= 0) continue; // padding slots
          xs.push(tNum);
          ys.push(fRel * fg);
          sizes.push(4 + 18 * (amp / maxAmps[c]));
          customs.push(amp);
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
  }, [initData, pairsHistory, timeSteps]);

  const peakLayout: Partial<Plotly.Layout> = {
    title: { text: "Peak-pair stream — frequency vs time, marker size ∝ amplitude", font: { size: 13 } },
    xaxis: { title: { text: "Time step" }, range: initData ? [0, initData.total_steps] : undefined },
    yaxis: {
      title: { text: "Frequency (Hz)" },
      range: initData
        ? [0, (initData.f_max_rel ?? 12) * initData.spindle_freq]
        : undefined,
    },
    legend: { orientation: "h", y: -0.2 },
    margin: { t: 40, r: 20, b: 60, l: 60 },
    height: 360,
  };
  const peakRef = usePlotly(peakScatterData, peakLayout);

  // -- Latest spectrum snapshot: stem plot of current K peaks per channel --
  const latestPairs = pairsHistory[pairsHistory.length - 1];
  const snapshotData: Plotly.Data[] = useMemo(() => {
    if (!initData || !latestPairs) return [];
    const fg = initData.spindle_freq;
    const traces: Plotly.Data[] = [];
    for (let c = 0; c < initData.n_channels; c++) {
      // Build vertical stems: for each peak emit (x, 0)->(x, amp) using None separators.
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
  }, [initData, latestPairs]);

  const snapshotLayout: Partial<Plotly.Layout> = {
    title: {
      text: latestPairs
        ? `Current spectrum (t=${timeSteps[timeSteps.length - 1]}) — top-${initData?.k_peaks ?? 0} peaks per channel`
        : "Current spectrum",
      font: { size: 13 },
    },
    xaxis: {
      title: { text: "Frequency (Hz)" },
      range: initData
        ? [0, (initData.f_max_rel ?? 12) * initData.spindle_freq]
        : undefined,
    },
    yaxis: { title: { text: "Amplitude" } },
    legend: { orientation: "h", y: -0.25 },
    margin: { t: 40, r: 20, b: 60, l: 60 },
    height: 280,
  };
  const snapshotRef = usePlotly(snapshotData, snapshotLayout);

  const selFileInfo = fileSource === "folders"
    ? files.find((f) => f.name === selFile)
    : testFiles.find((f) => f.path === selTestFile);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Simulation</h2>

      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Source">
            <select className={inputClass} value={fileSource} onChange={(e) => {
              const src = e.target.value as FileSource;
              setFileSource(src);
              if (src === "test_set") {
                getTestFiles().then((d) => {
                  setTestFiles(d.files || []);
                  if (d.files?.length && !selTestFile) setSelTestFile(d.files[0].path);
                });
              }
            }}>
              <option value="folders">All Folders</option>
              <option value="test_set">Test Set</option>
            </select>
          </Field>
          {fileSource === "folders" ? (
            <>
              <Field label="Folder">
                <select className={inputClass} value={selFolder} onChange={(e) => setSelFolder(e.target.value)}>
                  {folders.map((f) => (
                    <option key={f.name} value={f.name}>{f.name}</option>
                  ))}
                </select>
              </Field>
              <Field label="File">
                <select className={inputClass} value={selFile} onChange={(e) => setSelFile(e.target.value)}>
                  {files.map((f) => (
                    <option key={f.name} value={f.name}>
                      {f.name} {f.broke ? "🔴" : "🟢"}
                    </option>
                  ))}
                </select>
              </Field>
            </>
          ) : (
            <Field label="Test File">
              <select className={inputClass} value={selTestFile} onChange={(e) => setSelTestFile(e.target.value)}>
                {testFiles.map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.path} {f.broke ? "🔴" : "🟢"}
                  </option>
                ))}
              </select>
            </Field>
          )}
          <Field label={`Speed: ${speed} pts/s`}>
            <input
              type="range"
              min={1}
              max={200}
              value={speed}
              onChange={(e) => handleSpeedChange(+e.target.value)}
              className="w-32"
            />
          </Field>
          <div className="flex gap-2">
            {!running ? (
              <button className={btnPrimary} onClick={handleStart} disabled={fileSource === "folders" ? !selFile : !selTestFile}>
                ▶ Play
              </button>
            ) : paused ? (
              <button className={btnPrimary} onClick={handleResume}>▶ Resume</button>
            ) : (
              <button className={btnSecondary} onClick={handlePause}>⏸ Pause</button>
            )}
            {running && (
              <button className={btnDanger} onClick={handleStop}>⏹ Stop</button>
            )}
            {!running && timeSteps.length > 0 && (
              <button className={btnSecondary} onClick={reset}>Reset</button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 mt-3 pt-3 border-t border-gray-100 text-sm">
          {selFileInfo && (
            <>
              <Badge color={selFileInfo.broke ? "red" : "green"}>
                {selFileInfo.broke ? "Broke" : "OK"}
              </Badge>
              <span className="text-gray-500">{selFileInfo.n_samples.toLocaleString()} raw samples</span>
            </>
          )}
          {initData && (
            <>
              <span className="text-gray-500">{initData.total_steps} steps</span>
              <span className="text-gray-500">fg={initData.spindle_freq.toFixed(1)} Hz</span>
              <span className="text-gray-500">K={initData.k_peaks} peaks/ch</span>
              <span className="text-gray-500">channels: {initData.channel_names.join(", ")}</span>
            </>
          )}
          {latestProb !== null && (
            <Badge color={latestProb > 0.5 ? "red" : "green"}>
              P(broke) = {latestProb.toFixed(3)}
            </Badge>
          )}
          {avgInferenceMs !== null && (
            <span className="text-gray-500">⏱ avg {avgInferenceMs.toFixed(2)} ms/inference</span>
          )}
          {initData && (
            <span className="text-gray-400 ml-auto">{progress}%</span>
          )}
        </div>

        {initData && (
          <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
            <div
              className="bg-indigo-600 h-1.5 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </Card>

      <Card>
        <div ref={probRef} className="plotly-chart" />
      </Card>

      <Card>
        <div ref={peakRef} className="plotly-chart" />
      </Card>

      <Card>
        <div ref={snapshotRef} className="plotly-chart" />
      </Card>
    </div>
  );
}
