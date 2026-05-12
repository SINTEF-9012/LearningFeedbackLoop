import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type Plotly from "plotly.js";
import type { PipelineConfig, FolderInfo, LRStage, TrainStatus } from "../types";
import { getFolders, startTraining, getTrainStatus, stopTraining, resetModel, setConfig, getModelWeights, type ModelWeights } from "../api";
import { Card, Field, inputClass, btnPrimary, btnDanger, btnSecondary, Badge, usePlotly } from "../ui";

interface Props {
  config: PipelineConfig;
  onModelTrained: () => void;
  onModelReset: () => void;
}

export default function TrainingPanel({ config, onModelTrained, onModelReset }: Props) {
  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [selectedFolders, setSelectedFolders] = useState<Set<string>>(new Set());
  const [testSplit, setTestSplit] = useState(0.2);
  const [valSplit, setValSplit] = useState(0.15);
  const [batchSize, setBatchSize] = useState(16);
  const [patience, setPatience] = useState(0);
  const [nWindows, setNWindows] = useState(1);
  const [lrSchedule, setLrSchedule] = useState<LRStage[]>([
    { lr: 0.001, epochs: 200 },
    { lr: 0.0001, epochs: 500 },
  ]);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [info, setInfo] = useState<any>(null);
  const [error, setError] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  // Load folders
  useEffect(() => {
    getFolders().then((d) => {
      setFolders(d.folders || []);
      if (d.folders?.length) {
        setSelectedFolders(new Set(d.folders.map((f: FolderInfo) => f.name)));
      }
    });
  }, []);

  // Poll training status
  useEffect(() => {
    if (status?.running) {
      pollRef.current = setInterval(async () => {
        const s = await getTrainStatus();
        setStatus(s);
        if (!s.running) {
          clearInterval(pollRef.current);
          onModelTrained();
        }
      }, 500);
    }
    return () => clearInterval(pollRef.current);
  }, [status?.running, onModelTrained]);

  const toggleFolder = (name: string) => {
    setSelectedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const handleTrain = async () => {
    setError("");
    // Push config to server first
    const cfgRes = await setConfig(config);
    if (cfgRes.error) {
      setError(cfgRes.error);
      return;
    }
    const res = await startTraining({
      folders: Array.from(selectedFolders),
      test_split: testSplit,
      val_split: valSplit,
      lr_schedule: lrSchedule,
      batch_size: batchSize,
      patience,
      n_windows: nWindows,
    });
    if (res.error) {
      setError(res.error);
      return;
    }
    setInfo(res);
    // When continuing, keep the existing history from the status
    const prevHistory = status?.history ?? [];
    const prevValHistory = status?.val_history ?? [];
    setStatus({
      running: true,
      current_epoch: prevHistory.length,
      total_epochs: 0,
      current_stage: 0,
      total_stages: 0,
      history: prevHistory,
      val_history: prevValHistory,
      best_val_loss: status?.best_val_loss ?? null,
      best_epoch: status?.best_epoch ?? -1,
      epochs_since_improve: status?.epochs_since_improve ?? 0,
      early_stopped: false,
    });
  };

  const handleStop = () => stopTraining();

  const handleReset = async () => {
    await resetModel();
    setStatus(null);
    setInfo(null);
    setError("");
    onModelReset();
  };

  const updateLR = (idx: number, field: keyof LRStage, val: number) => {
    setLrSchedule((prev) => prev.map((s, i) => (i === idx ? { ...s, [field]: val } : s)));
  };

  const addStage = () => setLrSchedule((p) => [...p, { lr: 0.00001, epochs: 100 }]);
  const removeStage = (idx: number) => setLrSchedule((p) => p.filter((_, i) => i !== idx));

  // Loss curve
  const lossData: Plotly.Data[] = [];
  if (status?.history?.length) {
    lossData.push({ y: status.history, type: "scatter", mode: "lines", line: { color: "#4f46e5", width: 1.5 }, name: "Train" });
  }
  if (status?.val_history?.length) {
    lossData.push({ y: status.val_history, type: "scatter", mode: "lines", line: { color: "#f59e0b", width: 1.5 }, name: "Validation" });
  }
  if (status?.best_epoch && status.best_epoch > 0 && status.best_val_loss != null) {
    lossData.push({
      x: [status.best_epoch - 1],
      y: [status.best_val_loss],
      type: "scatter",
      mode: "markers",
      marker: { color: "#10b981", size: 10, symbol: "star", line: { color: "#065f46", width: 1 } },
      name: "Best val",
    });
  }
  const lossLayout: Partial<Plotly.Layout> = {
    title: { text: "Training Loss", font: { size: 13 } },
    xaxis: { title: { text: "Epoch" } },
    yaxis: { title: { text: "BCE Loss" } },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 300,
    legend: { orientation: "h", y: -0.2 },
  };
  const lossRef = usePlotly(lossData, lossLayout);

  // -- Pair-encoder weights visualization -----------------------------------
  const [weights, setWeights] = useState<ModelWeights | null>(null);
  const refreshWeights = useCallback(async () => {
    const w = await getModelWeights();
    if (!w.error) setWeights(w);
  }, []);
  // Auto-refresh weights when training transitions from running -> stopped.
  const wasRunningRef = useRef(false);
  useEffect(() => {
    const running = status?.running ?? false;
    if (wasRunningRef.current && !running) {
      refreshWeights();
    }
    wasRunningRef.current = running;
  }, [status?.running, refreshWeights]);

  // Per-pair encoder first-layer weights: shape (D, 2). Each row is one
  // hidden neuron's reading of (f_rel, amp). Together they form the basis
  // the model uses to encode a single (frequency, amplitude) peak.
  const wHeatmapData: Plotly.Data[] = useMemo(() => {
    if (!weights) return [];
    const absMax = weights.pair_encoder_W1.reduce(
      (m, row) => Math.max(m, ...row.map((v) => Math.abs(v))), 0
    ) || 1;
    return [{
      z: weights.pair_encoder_W1,
      x: weights.pair_input_labels,
      y: weights.pair_encoder_W1.map((_, i) => `n${i}`),
      type: "heatmap" as const,
      colorscale: "RdBu",
      reversescale: true,
      zmin: -absMax,
      zmax: absMax,
      colorbar: { title: { text: "W" } },
      hovertemplate: "neuron %{y} \u2190 %{x}<br>W = %{z:.4f}<extra></extra>",
    }];
  }, [weights]);

  const wHeatmapLayout: Partial<Plotly.Layout> = useMemo(() => ({
    title: { text: "Pair encoder W\u2081 \u2014 hidden neurons \u2190 (f_rel, amp)", font: { size: 13 } },
    xaxis: { title: { text: "Pair input" }, side: "top" },
    yaxis: { title: { text: "Hidden neuron" }, autorange: "reversed" },
    margin: { t: 60, r: 20, b: 40, l: 80 },
    height: weights ? Math.max(280, 14 * weights.pair_encoder_W1.length + 80) : 280,
  }), [weights]);
  const wHeatmapRef = usePlotly(wHeatmapData, wHeatmapLayout);

  // Sample the encoder over a small (f_rel, amp) grid to show what feature
  // each neuron responds to. We use only the first linear layer + ReLU as
  // a quick sketch of selectivity; later layers compose these.
  const encoderResponseData: Plotly.Data[] = useMemo(() => {
    if (!weights) return [];
    const W1 = weights.pair_encoder_W1;
    const b1 = weights.pair_encoder_b1;
    const D = W1.length;
    // Pick a few representative neurons (first 6 by output norm of W1 row)
    const norms = W1.map((row, i) => ({ i, n: Math.hypot(row[0], row[1]) }));
    norms.sort((a, b) => b.n - a.n);
    const picks = norms.slice(0, Math.min(6, D)).map((p) => p.i);
    // For each picked neuron, sample over f_rel ∈ [0.5, 12], amp normalized 1.
    const fGrid = Array.from({ length: 60 }, (_, i) => 0.5 + i * (12 - 0.5) / 59);
    const palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4"];
    return picks.map((idx, k) => ({
      x: fGrid,
      y: fGrid.map((f) => Math.max(0, W1[idx][0] * f + W1[idx][1] * 1.0 + b1[idx])),
      type: "scatter" as const,
      mode: "lines" as const,
      name: `n${idx}`,
      line: { color: palette[k % palette.length], width: 1.5 },
    }));
  }, [weights]);

  const encoderResponseLayout: Partial<Plotly.Layout> = {
    title: { text: "Encoder neuron response (amp = 1, varying f_rel) \u2014 ReLU(W\u2081·[f_rel, 1] + b\u2081)", font: { size: 13 } },
    xaxis: { title: { text: "f_rel = f / fg" } },
    yaxis: { title: { text: "Activation" }, zeroline: true, zerolinecolor: "#9ca3af" },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 320,
    legend: { orientation: "h", y: -0.25 },
  };
  const wColumnsRef = usePlotly(encoderResponseData, encoderResponseLayout);

  const isTraining = status?.running ?? false;
  const progress = status ? Math.round((status.current_epoch / Math.max(status.total_epochs, 1)) * 100) : 0;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Training</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Folder selection */}
        <Card title="Data Folders" className="lg:col-span-1">
          <div className="space-y-1 max-h-64 overflow-auto">
            {folders.map((f) => (
              <label key={f.name} className="flex items-center gap-2 text-sm cursor-pointer hover:bg-gray-50 px-1 py-0.5 rounded">
                <input
                  type="checkbox"
                  checked={selectedFolders.has(f.name)}
                  onChange={() => toggleFolder(f.name)}
                  className="rounded text-indigo-600"
                />
                <span className="flex-1 truncate">{f.name}</span>
                <span className="text-xs text-gray-400">{f.n_files} files</span>
              </label>
            ))}
            {folders.length === 0 && (
              <p className="text-sm text-gray-400">No folders found. Check data directory in Config.</p>
            )}
          </div>
        </Card>

        {/* Training parameters */}
        <Card title="Training Parameters" className="lg:col-span-1">
          <div className="space-y-3">
            <Field label={`Test Split: ${(testSplit * 100).toFixed(0)}%`}>
              <input
                type="range"
                min={0.1}
                max={0.5}
                step={0.05}
                value={testSplit}
                onChange={(e) => setTestSplit(+e.target.value)}
                className="w-full"
              />
            </Field>
            <Field label={`Val Split (of train): ${(valSplit * 100).toFixed(0)}%`}>
              <input
                type="range"
                min={0}
                max={0.4}
                step={0.05}
                value={valSplit}
                onChange={(e) => setValSplit(+e.target.value)}
                className="w-full"
              />
            </Field>
            <Field label="Early-Stop Patience (epochs, 0 = off)">
              <input
                type="number"
                min={0}
                className={inputClass}
                value={patience}
                onChange={(e) => setPatience(Math.max(0, +e.target.value))}
              />
            </Field>
            <Field label="Batch Size">
              <input type="number" className={inputClass} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} />
            </Field>
            <Field label="Windows per sample / epoch">
              <input
                type="number"
                min={1}
                className={inputClass}
                value={nWindows}
                onChange={(e) => setNWindows(Math.max(1, +e.target.value))}
              />
            </Field>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-600">LR Schedule</span>
                <button onClick={addStage} className="text-xs text-indigo-600 hover:text-indigo-800">+ Add stage</button>
              </div>
              {lrSchedule.map((stage, i) => (
                <div key={i} className="flex items-center gap-2">
                  <input
                    type="number"
                    step="any"
                    className={`${inputClass} w-28`}
                    value={stage.lr}
                    onChange={(e) => updateLR(i, "lr", +e.target.value)}
                    placeholder="lr"
                  />
                  <span className="text-xs text-gray-400">×</span>
                  <input
                    type="number"
                    className={`${inputClass} w-20`}
                    value={stage.epochs}
                    onChange={(e) => updateLR(i, "epochs", +e.target.value)}
                    placeholder="epochs"
                  />
                  {lrSchedule.length > 1 && (
                    <button onClick={() => removeStage(i)} className="text-red-400 hover:text-red-600 text-sm">×</button>
                  )}
                </div>
              ))}
            </div>

            <div className="flex gap-2 pt-2">
              <button className={btnPrimary} onClick={handleTrain} disabled={isTraining || selectedFolders.size === 0}>
                {isTraining ? "Training…" : "Train Model"}
              </button>
              {isTraining && (
                <button className={btnDanger} onClick={handleStop}>Stop</button>
              )}
              <button className={btnSecondary} onClick={handleReset} disabled={isTraining}>
                Reset Model
              </button>
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}
          </div>
        </Card>

        {/* Status */}
        <Card title="Status" className="lg:col-span-1">
          {info && (
            <div className="space-y-1 text-sm text-gray-600 mb-3">
              <p>Train: <strong>{info.n_train}</strong> ({info.n_broke_train} broke)</p>
              {info.n_val > 0 && (
                <p>Val: <strong>{info.n_val}</strong> ({info.n_broke_val} broke)</p>
              )}
              <p>Test: <strong>{info.n_test}</strong> ({info.n_broke_test} broke)</p>
              <p>Parameters: <strong>{info.n_params?.toLocaleString()}</strong></p>
              <p>Device: <Badge color="blue">{info.device}</Badge></p>
            </div>
          )}
          {status && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Badge color={isTraining ? "yellow" : "green"}>
                  {isTraining ? "Training" : "Done"}
                </Badge>
                <span className="text-gray-500">
                  Stage {status.current_stage}/{status.total_stages} · Epoch {status.current_epoch}/{status.total_epochs}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              {status.history.length > 0 && (
                <p className="text-xs text-gray-500">
                  Latest train loss: {status.history[status.history.length - 1].toFixed(4)}
                  {status.val_history?.length > 0 && (
                    <> · val: {status.val_history[status.val_history.length - 1].toFixed(4)}</>
                  )}
                </p>
              )}
              {status.best_val_loss !== null && status.best_val_loss !== undefined && (
                <p className="text-xs text-emerald-700">
                  Best val: <strong>{status.best_val_loss.toFixed(4)}</strong> @ epoch {status.best_epoch}
                  {patience > 0 && status.running && (
                    <> · no improvement for {status.epochs_since_improve}/{patience}</>
                  )}
                </p>
              )}
              {status.early_stopped && !status.running && (
                <p className="text-xs text-amber-700">
                  Early stopped — best weights restored.
                </p>
              )}
              {status.best_val_loss !== null && status.best_val_loss !== undefined &&
                !status.running && !status.early_stopped && (
                  <p className="text-xs text-emerald-700">
                    Best-validation weights restored.
                  </p>
                )}
            </div>
          )}
        </Card>
      </div>

      {/* Loss curve */}
      <Card title="Loss Curve">
        <div ref={lossRef} className="plotly-chart" />
      </Card>

      {/* Pair-encoder weights visualization */}
      <Card title="Pair encoder (per-peak shared MLP)">
        {!weights ? (
          <div className="space-y-3">
            <p className="text-sm text-gray-400 py-2 text-center">
              Train a model to inspect the per-pair encoder weights.
            </p>
            <div className="flex justify-center">
              <button
                onClick={refreshWeights}
                className="text-xs text-indigo-600 hover:text-indigo-800"
                disabled={isTraining}
              >
                Try fetch
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-4">
              <p className="text-xs text-gray-500 flex-1">
                Each (frequency, amplitude) peak is fed through a shared MLP. The first
                layer has weights of shape ({weights.pair_encoder_W1.length} hidden ×{" "}
                {weights.pair_input_labels.length} inputs). The heatmap shows raw
                weights; the lower chart sweeps each strong neuron over a range of
                f<sub>rel</sub> with amplitude fixed at 1, so you can see which
                neurons are tuned to which spectral region.
              </p>
              <button
                onClick={refreshWeights}
                className="text-xs text-indigo-600 hover:text-indigo-800 flex-shrink-0"
                disabled={isTraining}
              >
                Refresh
              </button>
            </div>
            <div ref={wHeatmapRef} className="plotly-chart" />
            <div ref={wColumnsRef} className="plotly-chart" />
            <div className="text-xs text-gray-500 pt-2 border-t border-gray-100">
              <p className="font-medium mb-1">Cutting-parameter standardization</p>
              <table className="text-xs w-full">
                <thead>
                  <tr className="text-gray-400">
                    <th className="text-left">param</th>
                    <th className="text-right">mean</th>
                    <th className="text-right">std</th>
                  </tr>
                </thead>
                <tbody>
                  {weights.param_keys.map((k, i) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td className="text-right tabular-nums">{weights.param_mean[i].toFixed(4)}</td>
                      <td className="text-right tabular-nums">{weights.param_std[i].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
