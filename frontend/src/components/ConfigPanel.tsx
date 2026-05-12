import { useState } from "react";
import type { PipelineConfig } from "../types";
import { setConfig as saveConfig } from "../api";
import { Card, Field, inputClass, btnPrimary, Badge } from "../ui";

interface Props {
  config: PipelineConfig;
  onChange: (c: PipelineConfig) => void;
}

export default function ConfigPanel({ config, onChange }: Props) {
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  const update = <K extends keyof PipelineConfig>(key: K, val: PipelineConfig[K]) => {
    onChange({ ...config, [key]: val });
  };

  const save = async () => {
    setSaving(true);
    setMsg("");
    const res = await saveConfig(config);
    if (res.error) setMsg(`Error: ${res.error}`);
    else setMsg("Saved");
    setSaving(false);
    setTimeout(() => setMsg(""), 3000);
  };

  const tOut = Math.floor(config.cnn_window / 2 ** config.conv_channels.length);
  const perStepDim = 2 * config.pair_embed_dim; // n_channels=2 (X,Y)

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Pipeline Configuration</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Data">
          <Field label="Data Directory">
            <input
              className={inputClass}
              value={config.data_dir}
              onChange={(e) => update("data_dir", e.target.value)}
            />
          </Field>
          <Field label="Sample rate (Hz)">
            <input
              type="number"
              className={inputClass}
              value={config.sample_rate}
              onChange={(e) => update("sample_rate", +e.target.value)}
            />
          </Field>
        </Card>

        <Card title="Peak extraction (FFT)">
          <div className="space-y-2">
            <Field label="FFT window (samples)">
              <input
                type="number"
                className={inputClass}
                value={config.fft_window}
                onChange={(e) => update("fft_window", +e.target.value)}
              />
            </Field>
            <Field label="FFT stride (samples)">
              <input
                type="number"
                className={inputClass}
                value={config.fft_step}
                onChange={(e) => update("fft_step", +e.target.value)}
              />
            </Field>
            <Field label="Peaks per channel (K)">
              <input
                type="number"
                className={inputClass}
                value={config.k_peaks}
                onChange={(e) => update("k_peaks", +e.target.value)}
              />
            </Field>
            <Field label="Max frequency (× spindle, blank = no cap)">
              <input
                className={inputClass}
                value={config.f_max_rel ?? ""}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  update("f_max_rel", v === "" ? null : +v);
                }}
              />
            </Field>
          </div>
        </Card>

        <Card title="Model architecture">
          <div className="space-y-2">
            <Field label="Pair encoder embedding dim (D)">
              <input
                type="number"
                className={inputClass}
                value={config.pair_embed_dim}
                onChange={(e) => update("pair_embed_dim", +e.target.value)}
              />
            </Field>
            <Field label="CNN window (time steps)">
              <input
                type="number"
                className={inputClass}
                value={config.cnn_window}
                onChange={(e) => update("cnn_window", +e.target.value)}
              />
            </Field>
            <Field label="Conv channels (comma-separated)">
              <input
                className={inputClass}
                value={config.conv_channels.join(", ")}
                onChange={(e) =>
                  update(
                    "conv_channels",
                    e.target.value.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n))
                  )
                }
              />
            </Field>
            <Field label="FC hidden size">
              <input
                type="number"
                className={inputClass}
                value={config.fc_hidden}
                onChange={(e) => update("fc_hidden", +e.target.value)}
              />
            </Field>
            <Field label="Kernel size">
              <input
                type="number"
                className={inputClass}
                value={config.kernel_size}
                onChange={(e) => update("kernel_size", +e.target.value)}
              />
            </Field>
          </div>
        </Card>

        <Card title="Computed">
          <div className="space-y-1 text-sm text-gray-600">
            <p>Pairs per timestep: <strong>{2 * config.k_peaks}</strong> (2 channels × {config.k_peaks})</p>
            <p>Per-step embedding (after sum over K): <strong>{perStepDim}</strong></p>
            <p>Pooling layers: <strong>{config.conv_channels.length}</strong></p>
            <p>Temporal dim after pooling: <strong>{tOut}</strong></p>
            <p className="text-xs text-gray-400 pt-2">
              Each FFT window yields the top-{config.k_peaks} spectral peaks per channel as
              (f<sub>rel</sub> = f<sub>Hz</sub> / f<sub>g</sub>, amplitude) pairs. The model
              encodes each pair through a shared MLP and sums them per channel before the
              temporal CNN.
            </p>
          </div>
        </Card>
      </div>

      <div className="flex items-center gap-3">
        <button className={btnPrimary} onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save Config"}
        </button>
        {msg && <Badge color={msg.startsWith("Error") ? "red" : "green"}>{msg}</Badge>}
      </div>
    </div>
  );
}
