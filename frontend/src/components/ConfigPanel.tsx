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
        </Card>

        <Card title="FFT Parameters">
          <div className="space-y-2">
            <Field label="FFT Window Size (samples)">
              <input
                type="number"
                className={inputClass}
                value={config.fft_window}
                onChange={(e) => update("fft_window", +e.target.value)}
              />
            </Field>
            <Field label="FFT Step / Stride (samples)">
              <input
                type="number"
                className={inputClass}
                value={config.fft_step}
                onChange={(e) => update("fft_step", +e.target.value)}
              />
            </Field>
            <Field label="Harmonic Multipliers (comma-separated)">
              <input
                className={inputClass}
                value={config.harm_mults.join(", ")}
                onChange={(e) =>
                  update(
                    "harm_mults",
                    e.target.value
                      .split(",")
                      .map((s) => parseInt(s.trim()))
                      .filter((n) => !isNaN(n))
                  )
                }
              />
            </Field>
          </div>
        </Card>

        <Card title="CNN Architecture">
          <div className="space-y-2">
            <Field label="CNN Window Size (time steps)">
              <input
                type="number"
                className={inputClass}
                value={config.cnn_window}
                onChange={(e) => update("cnn_window", +e.target.value)}
              />
            </Field>
            <Field label="Conv Channels (comma-separated)">
              <input
                className={inputClass}
                value={config.conv_channels.join(", ")}
                onChange={(e) =>
                  update(
                    "conv_channels",
                    e.target.value
                      .split(",")
                      .map((s) => parseInt(s.trim()))
                      .filter((n) => !isNaN(n))
                  )
                }
              />
            </Field>
            <Field label="FC Hidden Size">
              <input
                type="number"
                className={inputClass}
                value={config.fc_hidden}
                onChange={(e) => update("fc_hidden", +e.target.value)}
              />
            </Field>
            <Field label="Kernel Size">
              <input
                type="number"
                className={inputClass}
                value={config.kernel_size}
                onChange={(e) => update("kernel_size", +e.target.value)}
              />
            </Field>
          </div>
        </Card>

        <Card title="Computed Info">
          <div className="space-y-1 text-sm text-gray-600">
            <p>Harmonic features per step: <strong>{config.harm_mults.length * 3}</strong> ({config.harm_mults.length} harmonics × 3 channels)</p>
            <p>Pooling layers: <strong>{config.conv_channels.length}</strong></p>
            <p>Temporal dim after pooling: <strong>{Math.floor(config.cnn_window / (2 ** config.conv_channels.length))}</strong></p>
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
