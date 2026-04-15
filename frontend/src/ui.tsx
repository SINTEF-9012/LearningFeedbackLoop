import { useEffect, useRef, useCallback } from "react";
import Plotly from "plotly.js-dist-min";

/** Thin wrapper around Plotly.react for use in React components. */
export function usePlotly(
  data: Plotly.Data[],
  layout: Partial<Plotly.Layout>,
  config?: Partial<Plotly.Config>
) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    Plotly.react(ref.current, data, layout as Plotly.Layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      ...config,
    } as Plotly.Config);
  }, [data, layout, config]);

  // Resize on mount
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(() => Plotly.Plots.resize(el));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return ref;
}

/** Simple card wrapper. */
export function Card({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {title && (
        <div className="px-4 py-3 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-700">{title}</h3>
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

/** Labeled input field. */
export function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block text-xs font-medium text-gray-600 mb-1">
      {label}
      <div className="mt-0.5">{children}</div>
    </label>
  );
}

/** Status badge. */
export function Badge({
  color,
  children,
}: {
  color: "green" | "red" | "gray" | "blue" | "yellow";
  children: React.ReactNode;
}) {
  const colors = {
    green: "bg-emerald-100 text-emerald-700",
    red: "bg-red-100 text-red-700",
    gray: "bg-gray-100 text-gray-600",
    blue: "bg-blue-100 text-blue-700",
    yellow: "bg-yellow-100 text-yellow-700",
  };
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${colors[color]}`}>
      {children}
    </span>
  );
}

export const inputClass =
  "w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 outline-none";

export const btnPrimary =
  "px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors";

export const btnDanger =
  "px-4 py-2 bg-red-600 text-white text-sm font-medium rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors";

export const btnSecondary =
  "px-4 py-2 bg-gray-200 text-gray-700 text-sm font-medium rounded hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors";
