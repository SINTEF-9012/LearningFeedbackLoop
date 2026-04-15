import { useState, useEffect, useCallback } from "react";
import type Plotly from "plotly.js";
import type { PipelineConfig, FolderInfo, EvalResponse, EvalResult } from "../types";
import { getFolders, evaluate } from "../api";
import { Card, Field, inputClass, btnPrimary, btnSecondary, Badge, usePlotly } from "../ui";

interface Props {
  config: PipelineConfig;
}

export default function EvaluationPanel({ config }: Props) {
  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [selectedFolders, setSelectedFolders] = useState<Set<string>>(new Set());
  const [source, setSource] = useState<"test_set" | "folders">("test_set");
  const [windowPos, setWindowPos] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getFolders().then((d) => {
      setFolders(d.folders || []);
      if (d.folders?.length) setSelectedFolders(new Set(d.folders.map((f: FolderInfo) => f.name)));
    });
  }, []);

  const toggleFolder = (name: string) => {
    setSelectedFolders((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const handleEval = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    const res = await evaluate({
      source,
      folders: source === "folders" ? Array.from(selectedFolders) : [],
      window_position: windowPos,
    });
    if (res.error) {
      setError(res.error);
    } else {
      setResult(res);
    }
    setLoading(false);
  };

  // Confusion matrix heatmap
  const cm = result?.confusion_matrix;
  const cmData: Plotly.Data[] = cm
    ? [
        {
          z: cm,
          x: ["Pred OK", "Pred Broke"],
          y: ["True OK", "True Broke"],
          type: "heatmap",
          colorscale: [
            [0, "#f0fdf4"],
            [1, "#4f46e5"],
          ],
          hovertemplate: "%{y} / %{x}: %{z}<extra></extra>",
          showscale: false,
        } as any,
      ]
    : [];
  const cmLayout: Partial<Plotly.Layout> = {
    title: { text: "Confusion Matrix", font: { size: 13 } },
    xaxis: { side: "bottom" },
    yaxis: { autorange: "reversed" },
    margin: { t: 40, r: 20, b: 60, l: 80 },
    height: 280,
    width: 350,
  };
  const cmRef = usePlotly(cmData, cmLayout);

  // Probability distribution
  const probData: Plotly.Data[] = result
    ? [
        {
          x: result.results.filter((r) => r.true_label === 0).map((r) => r.probability),
          type: "histogram",
          name: "OK",
          marker: { color: "#10b981" },
          opacity: 0.7,
        } as any,
        {
          x: result.results.filter((r) => r.true_label === 1).map((r) => r.probability),
          type: "histogram",
          name: "Broke",
          marker: { color: "#ef4444" },
          opacity: 0.7,
        } as any,
      ]
    : [];
  const probLayout: Partial<Plotly.Layout> = {
    title: { text: "Probability Distribution", font: { size: 13 } },
    barmode: "overlay",
    xaxis: { title: { text: "P(broke)" }, range: [0, 1] },
    yaxis: { title: { text: "Count" } },
    margin: { t: 40, r: 20, b: 50, l: 50 },
    height: 280,
    legend: { orientation: "h", y: -0.25 },
  };
  const probRef = usePlotly(probData, probLayout);

  const report = result?.classification_report;

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Evaluation</h2>

      {/* Controls */}
      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Source">
            <select className={inputClass} value={source} onChange={(e) => setSource(e.target.value as any)}>
              <option value="test_set">Test Set (from training split)</option>
              <option value="folders">Select Folders</option>
            </select>
          </Field>
          <Field label={`Window Position: ${(windowPos * 100).toFixed(0)}%`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={windowPos}
              onChange={(e) => setWindowPos(+e.target.value)}
              className="w-32"
            />
          </Field>
          <button className={btnPrimary} onClick={handleEval} disabled={loading}>
            {loading ? "Evaluating…" : "Run Evaluation"}
          </button>
        </div>

        {source === "folders" && (
          <div className="mt-3 pt-3 border-t border-gray-100">
            <p className="text-xs font-medium text-gray-600 mb-1">Folders</p>
            <div className="flex flex-wrap gap-2">
              {folders.map((f) => (
                <label key={f.name} className="flex items-center gap-1.5 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedFolders.has(f.name)}
                    onChange={() => toggleFolder(f.name)}
                    className="rounded text-indigo-600"
                  />
                  {f.name}
                </label>
              ))}
            </div>
          </div>
        )}

        {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
      </Card>

      {/* Results */}
      {result && (
        <>
          {/* Summary */}
          <div className="flex flex-wrap gap-3">
            <Card className="flex-1 min-w-[140px]">
              <div className="text-center">
                <p className="text-3xl font-bold text-indigo-600">{(result.accuracy * 100).toFixed(1)}%</p>
                <p className="text-xs text-gray-500 mt-1">Accuracy</p>
              </div>
            </Card>
            <Card className="flex-1 min-w-[140px]">
              <div className="text-center">
                <p className="text-3xl font-bold">{result.n_samples}</p>
                <p className="text-xs text-gray-500 mt-1">Samples</p>
              </div>
            </Card>
            <Card className="flex-1 min-w-[140px]">
              <div className="text-center">
                <p className="text-3xl font-bold text-amber-600">{result.avg_inference_ms.toFixed(2)}</p>
                <p className="text-xs text-gray-500 mt-1">Avg Inference (ms)</p>
              </div>
            </Card>
            {report && (
              <>
                <Card className="flex-1 min-w-[140px]">
                  <div className="text-center">
                    <p className="text-3xl font-bold text-emerald-600">{((report["OK"]?.["f1-score"] ?? 0) * 100).toFixed(0)}%</p>
                    <p className="text-xs text-gray-500 mt-1">F1 (OK)</p>
                  </div>
                </Card>
                <Card className="flex-1 min-w-[140px]">
                  <div className="text-center">
                    <p className="text-3xl font-bold text-red-600">{((report["Broke"]?.["f1-score"] ?? 0) * 100).toFixed(0)}%</p>
                    <p className="text-xs text-gray-500 mt-1">F1 (Broke)</p>
                  </div>
                </Card>
              </>
            )}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <div ref={cmRef} className="plotly-chart" />
            </Card>
            <Card>
              <div ref={probRef} className="plotly-chart" />
            </Card>
          </div>

          {/* Classification report table */}
          {report && (
            <Card title="Classification Report">
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 text-left text-xs text-gray-500 uppercase">
                      <th className="py-2 pr-4">Class</th>
                      <th className="py-2 pr-4">Precision</th>
                      <th className="py-2 pr-4">Recall</th>
                      <th className="py-2 pr-4">F1-Score</th>
                      <th className="py-2">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["OK", "Broke"].map((cls) => {
                      const r = report[cls];
                      if (!r) return null;
                      return (
                        <tr key={cls} className="border-b border-gray-100">
                          <td className="py-2 pr-4 font-medium">{cls}</td>
                          <td className="py-2 pr-4">{(r.precision * 100).toFixed(1)}%</td>
                          <td className="py-2 pr-4">{(r.recall * 100).toFixed(1)}%</td>
                          <td className="py-2 pr-4">{(r["f1-score"] * 100).toFixed(1)}%</td>
                          <td className="py-2">{r.support}</td>
                        </tr>
                      );
                    })}
                    {report["weighted avg"] && (
                      <tr className="font-medium">
                        <td className="py-2 pr-4">Weighted Avg</td>
                        <td className="py-2 pr-4">{(report["weighted avg"].precision * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-4">{(report["weighted avg"].recall * 100).toFixed(1)}%</td>
                        <td className="py-2 pr-4">{(report["weighted avg"]["f1-score"] * 100).toFixed(1)}%</td>
                        <td className="py-2">{report["weighted avg"].support}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Per-sample results */}
          <Card title="Per-Sample Results">
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b border-gray-200 text-left text-xs text-gray-500 uppercase">
                    <th className="py-2 pr-4">File</th>
                    <th className="py-2 pr-4">True</th>
                    <th className="py-2 pr-4">Predicted</th>
                    <th className="py-2 pr-4">P(broke)</th>
                    <th className="py-2">Correct</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r, i) => {
                    const correct = r.true_label === r.predicted;
                    return (
                      <tr key={i} className={`border-b border-gray-50 ${!correct ? "bg-red-50" : ""}`}>
                        <td className="py-1.5 pr-4 font-mono text-xs">{r.file}</td>
                        <td className="py-1.5 pr-4">
                          <Badge color={r.true_label ? "red" : "green"}>
                            {r.true_label ? "Broke" : "OK"}
                          </Badge>
                        </td>
                        <td className="py-1.5 pr-4">
                          <Badge color={r.predicted ? "red" : "green"}>
                            {r.predicted ? "Broke" : "OK"}
                          </Badge>
                        </td>
                        <td className="py-1.5 pr-4">{r.probability.toFixed(4)}</td>
                        <td className="py-1.5">{correct ? "✓" : "✗"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
