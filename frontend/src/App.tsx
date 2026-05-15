import { useState, useEffect, useCallback } from "react";
import type { PipelineConfig } from "./types";
import { defaultConfig } from "./types";
import { getConfig } from "./api";
import ConfigPanel from "./components/ConfigPanel";
import TrainingPanel from "./components/TrainingPanel";
import SimulationPanel from "./components/SimulationPanel";
import EvaluationPanel from "./components/EvaluationPanel";
import OFReplayPanel from "./components/OFReplayPanel";

type Tab = "training" | "simulation" | "evaluation" | "of_replay" | "config";

const tabs: { id: Tab; label: string }[] = [
  { id: "training", label: "Training" },
  { id: "simulation", label: "Simulation" },
  { id: "evaluation", label: "Evaluation" },
  { id: "of_replay", label: "OF Replay" },
  { id: "config", label: "Config" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("training");
  const [config, setConfig] = useState<PipelineConfig>(defaultConfig);
  const [modelReady, setModelReady] = useState(false);
  const [device, setDevice] = useState("");

  // Load server config on mount
  useEffect(() => {
    getConfig().then((c) => {
      setConfig({
        data_dir: c.data_dir,
        fft_window: c.fft_window,
        fft_step: c.fft_step,
        sample_rate: c.sample_rate,
        k_peaks: c.k_peaks,
        f_max_rel: c.f_max_rel,
        cnn_window: c.cnn_window,
        pair_embed_dim: c.pair_embed_dim,
        conv_channels: c.conv_channels,
        fc_hidden: c.fc_hidden,
        kernel_size: c.kernel_size,
      });
      setModelReady(c.model_loaded);
      setDevice(c.device);
    }).catch(() => {});
  }, []);

  const handleModelTrained = useCallback(() => setModelReady(true), []);
  const handleModelReset = useCallback(() => setModelReady(false), []);

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-56 bg-slate-900 text-white flex flex-col flex-shrink-0">
        <div className="p-4 border-b border-slate-700">
          <h1 className="text-lg font-bold tracking-tight">ToolBreak</h1>
          <p className="text-xs text-slate-400 mt-0.5">Harmonic Pipeline GUI</p>
        </div>
        <nav className="flex-1 p-2 space-y-0.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
                tab === t.id
                  ? "bg-indigo-600 font-medium"
                  : "text-slate-300 hover:bg-slate-700/60"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="p-4 border-t border-slate-700 space-y-2">
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full flex-shrink-0 ${
                modelReady ? "bg-emerald-400" : "bg-slate-500"
              }`}
            />
            <span className="text-slate-400">
              {modelReady ? "Model loaded" : "No model"}
            </span>
          </div>
          {device && (
            <div className="text-xs text-slate-500">Device: {device}</div>
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-7xl mx-auto p-6 space-y-4">
          {/* All panels stay mounted so their internal state (training progress,
              plots, form inputs, simulation playback) persists across tab switches.
              Inactive panels are hidden with CSS instead of unmounted. */}
          <div hidden={tab !== "config"}>
            <ConfigPanel config={config} onChange={setConfig} />
          </div>
          <div hidden={tab !== "training"}>
            <TrainingPanel
              config={config}
              onModelTrained={handleModelTrained}
              onModelReset={handleModelReset}
            />
          </div>
          <div hidden={tab !== "simulation"}>
            <SimulationPanel config={config} />
          </div>
          <div hidden={tab !== "evaluation"}>
            <EvaluationPanel config={config} />
          </div>
          <div hidden={tab !== "of_replay"}>
            <OFReplayPanel config={config} />
          </div>
        </div>
      </main>
    </div>
  );
}
