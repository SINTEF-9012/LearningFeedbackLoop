import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type Plotly from "plotly.js";
import type { PipelineConfig, FolderInfo, FileInfo, SimInit, SimStep } from "../types";
import { getFolders, getFiles, getTestFiles } from "../api";
import type { TestFileInfo } from "../api";
import { Card, Field, inputClass, btnPrimary, btnDanger, btnSecondary, Badge, usePlotly } from "../ui";

type AxisChoice = "X" | "Y" | "Z" | "Mag";
type FileSource = "folders" | "test_set";

interface Props {
  config: PipelineConfig;
}

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

  // Accumulated data
  const [timeSteps, setTimeSteps] = useState<number[]>([]);
  const [combinedVals, setCombinedVals] = useState<number[]>([]);
  const [probVals, setProbVals] = useState<(number | null)[]>([]);
  const [harmData, setHarmData] = useState<number[][]>([]);
  const [magHarmData, setMagHarmData] = useState<number[][]>([]);
  const [inferenceTimes, setInferenceTimes] = useState<number[]>([]);

  // Harmonic display controls
  const [selectedAxis, setSelectedAxis] = useState<AxisChoice>("X");
  const [visibleHarmonics, setVisibleHarmonics] = useState<Set<number>>(new Set());

  const wsRef = useRef<WebSocket | null>(null);

  // Load folders and test files
  useEffect(() => {
    getFolders().then((d) => setFolders(d.folders || [])).catch(() => {});
    getTestFiles().then((d) => {
      setTestFiles(d.files || []);
      if (d.files?.length) setSelTestFile(d.files[0].path);
    }).catch(() => {});
  }, []);

  // Load files when folder changes
  useEffect(() => {
    if (!selFolder) return;
    getFiles(selFolder).then((d) => {
      setFiles(d.files || []);
      if (d.files?.length) setSelFile(d.files[0].name);
    });
  }, [selFolder]);

  // Set initial folder
  useEffect(() => {
    if (folders.length && !selFolder) setSelFolder(folders[0].name);
  }, [folders, selFolder]);

  // Initialize visible harmonics when initData arrives
  useEffect(() => {
    if (initData) {
      setVisibleHarmonics(new Set(initData.harm_mults.map((_: number, i: number) => i)));
    }
  }, [initData]);

  const reset = useCallback(() => {
    setTimeSteps([]);
    setCombinedVals([]);
    setProbVals([]);
    setHarmData([]);
    setMagHarmData([]);
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
      ws.send(
        JSON.stringify({
          action: "start",
          file_path: filePath,
          speed,
        })
      );
      setRunning(true);
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "init") {
        setInitData(msg as SimInit);
      } else if (msg.type === "step") {
        const step = msg as SimStep;
        setTimeSteps((p) => [...p, step.t]);
        setCombinedVals((p) => [...p, step.combined]);
        setProbVals((p) => [...p, step.prob]);
        setHarmData((p) => [...p, step.harmonics]);
        setMagHarmData((p) => [...p, step.mag_harmonics]);
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
    if (running) {
      wsRef.current?.send(JSON.stringify({ action: "set_speed", speed: newSpeed }));
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const toggleHarmonic = (idx: number) => {
    setVisibleHarmonics((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const selectAllHarmonics = () => {
    if (initData) setVisibleHarmonics(new Set(initData.harm_mults.map((_: number, i: number) => i)));
    else setVisibleHarmonics(new Set(config.harm_mults.map((_: number, i: number) => i)));
  };

  const selectNoneHarmonics = () => setVisibleHarmonics(new Set());

  // -- Derived data --
  const validProbs = probVals.filter((p): p is number => p !== null);
  const validProbSteps = timeSteps.filter((_, i) => probVals[i] !== null);

  const latestProb = validProbs.length > 0 ? validProbs[validProbs.length - 1] : null;
  const progress = initData ? Math.round((timeSteps.length / initData.total_steps) * 100) : 0;
  const avgInferenceMs = inferenceTimes.length > 0
    ? inferenceTimes.reduce((a, b) => a + b, 0) / inferenceTimes.length
    : null;

  const nHarm = initData?.harm_mults.length ?? config.harm_mults.length;
  const axisOffset = selectedAxis === "X" ? 0 : selectedAxis === "Y" ? 1 : selectedAxis === "Z" ? 2 : -1;

  const spindleHarmIdx = initData ? initData.harm_mults.indexOf(1) : -1;
  const toothPassMult = initData ? initData.z : 0;
  const toothPassHarmIdx = initData ? initData.harm_mults.indexOf(toothPassMult) : -1;

  // Combined + P(broke) chart
  const mainData: Plotly.Data[] = [
    {
      x: timeSteps,
      y: combinedVals,
      type: "scatter",
      mode: "lines",
      name: "w · harmonics",
      line: { color: "#3b82f6", width: 1.2 },
      yaxis: "y",
    },
    {
      x: validProbSteps,
      y: validProbs,
      type: "scatter",
      mode: "lines",
      name: "P(broke)",
      line: { color: "#111827", width: 2 },
      yaxis: "y2",
    },
    {
      x: initData ? [0, initData.total_steps] : [],
      y: [0.5, 0.5],
      type: "scatter",
      mode: "lines",
      name: "threshold",
      line: { color: "#9ca3af", width: 1, dash: "dash" },
      yaxis: "y2",
      showlegend: false,
    },
  ];

  const mainLayout: Partial<Plotly.Layout> = {
    title: { text: "Combined Signal & Break Probability", font: { size: 13 } },
    xaxis: { title: { text: "Time step" }, range: initData ? [0, initData.total_steps] : undefined },
    yaxis: { title: { text: "w · harmonics" }, side: "left" },
    yaxis2: {
      title: { text: "P(broke)" },
      side: "right",
      overlaying: "y",
      range: [-0.05, 1.05],
    },
    legend: { orientation: "h", y: -0.2 },
    margin: { t: 40, r: 60, b: 60, l: 60 },
    height: 350,
  };
  const mainRef = usePlotly(mainData, mainLayout);

  // Build harmonic traces based on selected axis and checked harmonics
  const harmTraces: Plotly.Data[] = useMemo(() => {
    if (!initData || harmData.length === 0) return [];
    const mults = initData.harm_mults;
    const traces: Plotly.Data[] = [];
    const visibleIdxs = Array.from(visibleHarmonics).sort((a, b) => a - b);

    for (const hi of visibleIdxs) {
      if (hi >= mults.length) continue;
      const mult = mults[hi];
      const isSpindle = hi === spindleHarmIdx;
      const isToothPass = hi === toothPassHarmIdx;
      const isBold = isSpindle || isToothPass;

      let label: string;
      let yValues: number[];

      if (selectedAxis === "Mag") {
        label = `Mag·${mult}×fg`;
        yValues = magHarmData.map((h) => h[hi] ?? 0);
      } else {
        const chIdx = axisOffset;
        const colIdx = chIdx * nHarm + hi;
        label = `${selectedAxis}·${mult}×fg`;
        yValues = harmData.map((h) => h[colIdx] ?? 0);
      }

      if (isSpindle) label += " (spindle)";
      if (isToothPass) label += " (tooth-pass)";

      traces.push({
        x: timeSteps,
        y: yValues,
        type: "scatter" as const,
        mode: "lines" as const,
        name: label,
        line: { width: isBold ? 2.5 : 1 },
      });
    }
    return traces;
  }, [initData, harmData, magHarmData, timeSteps, selectedAxis, visibleHarmonics, nHarm, axisOffset, spindleHarmIdx, toothPassHarmIdx]);

  const harmLayout: Partial<Plotly.Layout> = {
    title: { text: `Harmonic Magnitudes — ${selectedAxis === "Mag" ? "||accel||" : `Accel ${selectedAxis}`}`, font: { size: 13 } },
    xaxis: { title: { text: "Time step" }, range: initData ? [0, initData.total_steps] : undefined },
    yaxis: { title: { text: "|FFT|" } },
    legend: { orientation: "h", y: -0.25, font: { size: 9 } },
    margin: { t: 40, r: 20, b: 70, l: 60 },
    height: 350,
  };
  const harmRef = usePlotly(harmTraces, harmLayout);

  // W-vector bar charts (one per channel: X, Y, Z, Mag)
  const wBarAxes = ["X", "Y", "Z", "Mag"] as const;
  const axisColors = { X: "#3b82f6", Y: "#10b981", Z: "#f59e0b", Mag: "#8b5cf6" };

  const wBarData: Plotly.Data[][] = useMemo(() => {
    if (!initData?.w_vec) return [[], [], [], []];
    const nH = initData.harm_mults.length;
    const labels = initData.harm_mults.map((m) => `${m}×fg`);
    return wBarAxes.map((_, ai) => {
      const vals = initData.w_vec.slice(ai * nH, (ai + 1) * nH);
      return [{
        x: labels,
        y: vals,
        type: "bar" as const,
        marker: { color: vals.map((v) => v >= 0 ? axisColors[wBarAxes[ai]] : "#ef4444") },
      }];
    });
  }, [initData]);

  const wBarLayouts: Partial<Plotly.Layout>[] = wBarAxes.map((ax) => ({
    title: { text: `w vector — ${ax === "Mag" ? "||accel||" : `Accel ${ax}`}`, font: { size: 13 } },
    xaxis: { title: { text: "Harmonic" } },
    yaxis: { title: { text: "Weight" } },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 220,
    showlegend: false,
  }));

  const wBarRef0 = usePlotly(wBarData[0], wBarLayouts[0]);
  const wBarRef1 = usePlotly(wBarData[1], wBarLayouts[1]);
  const wBarRef2 = usePlotly(wBarData[2], wBarLayouts[2]);
  const wBarRef3 = usePlotly(wBarData[3], wBarLayouts[3]);
  const wBarRefs = [wBarRef0, wBarRef1, wBarRef2, wBarRef3];

  const selFileInfo = fileSource === "folders"
    ? files.find((f) => f.name === selFile)
    : testFiles.find((f) => f.path === selTestFile);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Simulation</h2>

      {/* Controls */}
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

        {/* Info bar */}
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
              <span className="text-gray-500">{initData.total_steps} harmonic steps</span>
              <span className="text-gray-500">fg={initData.spindle_freq.toFixed(1)} Hz</span>
              <span className="text-gray-500">z={initData.z}</span>
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

        {/* Progress bar */}
        {initData && (
          <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
            <div
              className="bg-indigo-600 h-1.5 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </Card>

      {/* Main chart */}
      <Card>
        <div ref={mainRef} className="plotly-chart" />
      </Card>

      {/* W-vector bar charts — always mount divs so refs stay attached */}
      <Card title="Learned Harmonic Weights (w = params × Wᵀ)">
        {!initData?.w_vec && (
          <p className="text-sm text-gray-400 py-4 text-center">
            Start a simulation to see the learned harmonic weights.
          </p>
        )}
        <div className={initData?.w_vec ? "space-y-2" : "hidden"}>
          {wBarAxes.map((ax, i) => (
            <div key={ax} ref={wBarRefs[i]} className="plotly-chart" />
          ))}
        </div>
      </Card>

      {/* Harmonics chart + control panel */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Harmonic selection panel */}
        <Card title="Harmonic Channels" className="lg:col-span-1">
          <div className="space-y-3">
            <Field label="Axis">
              <select
                className={inputClass}
                value={selectedAxis}
                onChange={(e) => setSelectedAxis(e.target.value as AxisChoice)}
              >
                <option value="X">Accel X</option>
                <option value="Y">Accel Y</option>
                <option value="Z">Accel Z</option>
                <option value="Mag">Magnitude (||accel||)</option>
              </select>
            </Field>

            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-gray-600">Harmonics</span>
                <span className="flex gap-1">
                  <button onClick={selectAllHarmonics} className="text-xs text-indigo-600 hover:text-indigo-800">All</button>
                  <span className="text-xs text-gray-300">|</span>
                  <button onClick={selectNoneHarmonics} className="text-xs text-indigo-600 hover:text-indigo-800">None</button>
                </span>
              </div>
              <div className="space-y-0.5 max-h-64 overflow-auto">
                {(initData?.harm_mults ?? config.harm_mults).map((mult, i) => {
                  const isSpindle = initData ? mult === 1 : false;
                  const isToothPass = initData ? mult === initData.z : false;
                  const isBold = isSpindle || isToothPass;
                  const freq = initData ? (initData.spindle_freq * mult).toFixed(0) : "?";
                  let suffix = "";
                  if (isSpindle) suffix = " · spindle";
                  if (isToothPass) suffix = " · tooth-pass";

                  return (
                    <label
                      key={mult}
                      className={`flex items-center gap-2 text-sm cursor-pointer hover:bg-gray-50 px-1 py-0.5 rounded ${
                        isBold ? "font-semibold" : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={visibleHarmonics.has(i)}
                        onChange={() => toggleHarmonic(i)}
                        className="rounded text-indigo-600"
                      />
                      <span className="flex-1">
                        {mult}×fg
                        <span className="text-xs text-gray-400 ml-1">({freq} Hz)</span>
                        {suffix && <span className="text-xs text-indigo-500 ml-1">{suffix}</span>}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>

        {/* Harmonic chart */}
        <Card className="lg:col-span-3">
          <div ref={harmRef} className="plotly-chart" />
        </Card>
      </div>
    </div>
  );
}
