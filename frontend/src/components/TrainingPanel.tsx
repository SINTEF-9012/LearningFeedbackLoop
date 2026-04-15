import { useState, useEffect, useRef, useCallback } from "react";
import type Plotly from "plotly.js";
import type { PipelineConfig, FolderInfo, LRStage, TrainStatus } from "../types";
import { getFolders, startTraining, getTrainStatus, stopTraining, resetModel, setConfig } from "../api";
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
  const [batchSize, setBatchSize] = useState(16);
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
      lr_schedule: lrSchedule,
      batch_size: batchSize,
    });
    if (res.error) {
      setError(res.error);
      return;
    }
    setInfo(res);
    // When continuing, keep the existing history from the status
    const prevHistory = status?.history ?? [];
    setStatus({ running: true, current_epoch: prevHistory.length, total_epochs: 0, current_stage: 0, total_stages: 0, history: prevHistory });
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
  const lossData: Plotly.Data[] = status?.history?.length
    ? [{ y: status.history, type: "scatter", mode: "lines", line: { color: "#4f46e5", width: 1.5 }, name: "Loss" }]
    : [];
  const lossLayout: Partial<Plotly.Layout> = {
    title: { text: "Training Loss", font: { size: 13 } },
    xaxis: { title: { text: "Epoch" } },
    yaxis: { title: { text: "BCE Loss" } },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 300,
  };
  const lossRef = usePlotly(lossData, lossLayout);

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
            <Field label="Batch Size">
              <input type="number" className={inputClass} value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} />
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
                  Latest loss: {status.history[status.history.length - 1].toFixed(4)}
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
    </div>
  );
}
