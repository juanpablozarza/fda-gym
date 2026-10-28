"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from "react";

export const CHART = {
  w: 640,
  h: 168,
  pad: { l: 36, r: 12, t: 12, b: 24 },
} as const;

export const DOT_LIMIT = 150;
export const BAR_LIMIT = 50;

const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

export function xAt(i: number, n: number, w: number = CHART.w, padL: number = CHART.pad.l, padR: number = CHART.pad.r): number {
  const innerW = w - padL - padR;
  return padL + (i / Math.max(n - 1, 1)) * innerW;
}

export function yAt(v: number, h: number = CHART.h, padT: number = CHART.pad.t, padB: number = CHART.pad.b): number {
  const innerH = h - padT - padB;
  const clamped = Math.min(1, Math.max(0, v));
  return padT + (1 - clamped) * innerH;
}

function svgCoords(svg: SVGSVGElement, clientX: number, clientY: number, vbW: number, vbH: number) {
  const rect = svg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return { x: 0, y: 0 };
  return {
    x: ((clientX - rect.left) / rect.width) * vbW,
    y: ((clientY - rect.top) / rect.height) * vbH,
  };
}

function pinFromPointer(
  event: PointerEvent<SVGSVGElement>,
  vbW: number,
  vbH: number,
  locate: (pt: { x: number; y: number }) => number,
  move: (i: number) => void,
  toggle: (i: number) => void,
) {
  if (event.button !== 0) return;
  const pt = svgCoords(event.currentTarget, event.clientX, event.clientY, vbW, vbH);
  const i = locate(pt);
  if (i < 0) return;
  move(i);
  toggle(i);
}

function nearestIndex(pts: { x: number; y: number }[], x: number, y: number): number {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].x - x;
    const dy = pts[i].y - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

export function usePinnedIndex(length: number) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const active = pinned ?? hover;

  useEffect(() => {
    const onDoc = (event: MouseEvent) => {
      if (!wrapRef.current?.contains(event.target as Node)) setPinned(null);
    };
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") setPinned(null);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  const move = (i: number) => setHover(i);
  const leave = (event: PointerEvent<HTMLElement>) => {
    if (!wrapRef.current?.contains(event.relatedTarget as Node)) setHover(null);
  };
  const toggle = (i: number) => setPinned((prev) => (prev === i ? null : i));
  const keyNav = (event: KeyboardEvent<SVGSVGElement | HTMLElement>) => {
    if (length <= 0) return;
    if (event.key === "Escape") {
      setPinned(null);
      return;
    }
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const cur = active ?? (event.key === "ArrowRight" ? -1 : length);
    const next = event.key === "ArrowRight" ? Math.min(length - 1, cur + 1) : Math.max(0, cur - 1);
    setHover(next);
    setPinned(next);
  };

  return { wrapRef, hover, pinned, active, move, leave, toggle, keyNav };
}

export function ChartTip({
  x,
  y,
  vbW,
  vbH,
  pinned,
  children,
}: {
  x: number;
  y: number;
  vbW: number;
  vbH: number;
  pinned: boolean;
  children: ReactNode;
}) {
  const flip = x / vbW > 0.62;
  return (
    <div
      className={`chart-tip glass${pinned ? " pinned" : ""}`}
      role="status"
      style={{
        left: `${(x / vbW) * 100}%`,
        top: `${(y / vbH) * 100}%`,
        transform: flip ? "translate(calc(-100% - 12px), -12px)" : "translate(12px, -12px)",
      }}
    >
      {children}
    </div>
  );
}

export function TipRow({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="tip-row">
      <span className="muted sm">{k}</span>
      <span className="mono sm">{v}</span>
    </div>
  );
}

function Grid({
  w,
  h,
  pad,
  ticks,
  floor,
}: {
  w: number;
  h: number;
  pad: { l: number; r: number; t: number; b: number };
  ticks: number[];
  floor?: number;
}) {
  const y = (v: number) => yAt(v, h, pad.t, pad.b);
  return (
    <>
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={pad.l} x2={w - pad.r} y1={y(tick)} y2={y(tick)} className="spark-grid" />
          <text x={pad.l - 8} y={y(tick) + 3} textAnchor="end" className="spark-tick">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {floor != null ? (
        <line x1={pad.l} x2={w - pad.r} y1={y(floor)} y2={y(floor)} className="spark-floor" />
      ) : null}
    </>
  );
}

export type LineSeries<T> = {
  id: string;
  color: string;
  points: { y: number; data: T }[];
};

export function NativeLine<T>({
  series,
  ariaLabel,
  floor,
  xStart,
  xEnd,
  liveText,
  renderTip,
  dotLimit = DOT_LIMIT,
}: {
  series: LineSeries<T>[];
  ariaLabel: string;
  floor?: number;
  xStart?: string;
  xEnd?: string;
  liveText?: (args: { i: number; series: LineSeries<T>; data: T }) => string;
  renderTip: (args: {
    i: number;
    y: number;
    color: string;
    series: LineSeries<T>;
    data: T;
    pinned: boolean;
  }) => ReactNode;
  dotLimit?: number;
}) {
  const w = CHART.w;
  const h = CHART.h;
  const pad = CHART.pad;
  const layout = useMemo(() => {
    const n = Math.max(2, ...series.map((s) => s.points.length));
    const hits: { x: number; y: number; si: number; pi: number }[] = [];
    series.forEach((s, si) => {
      s.points.forEach((p, pi) => {
        hits.push({ x: xAt(pi, n, w, pad.l, pad.r), y: yAt(p.y, h, pad.t, pad.b), si, pi });
      });
    });
    return { n, hits };
  }, [series, w, h, pad.l, pad.r, pad.t, pad.b]);

  const { wrapRef, pinned, active, move, leave, toggle, keyNav } = usePinnedIndex(layout.hits.length);
  const hit = active != null ? layout.hits[active] : undefined;
  const activeSeries = hit ? series[hit.si] : undefined;
  const activePoint = hit && activeSeries ? activeSeries.points[hit.pi] : undefined;

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (layout.hits.length === 0) return;
    const pt = svgCoords(event.currentTarget, event.clientX, event.clientY, w, h);
    move(nearestIndex(layout.hits, pt.x, pt.y));
  };

  return (
    <div className="chart-wrap" ref={wrapRef} onPointerLeave={leave}>
      <svg
        className="spark"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onPointerMove={onMove}
        onPointerDown={(event) =>
          pinFromPointer(
            event,
            w,
            h,
            (pt) => (layout.hits.length ? nearestIndex(layout.hits, pt.x, pt.y) : -1),
            move,
            toggle,
          )
        }
        onKeyDown={keyNav}
      >
        <Grid w={w} h={h} pad={pad} ticks={Y_TICKS} floor={floor} />
        {hit ? (
          <line x1={hit.x} x2={hit.x} y1={pad.t} y2={h - pad.b} className="spark-guide" />
        ) : null}
        {series.map((s) => {
          const n = layout.n;
          const d = s.points
            .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i, n, w, pad.l, pad.r).toFixed(1)} ${yAt(p.y, h, pad.t, pad.b).toFixed(1)}`)
            .join(" ");
          const showDots = s.points.length <= dotLimit;
          return (
            <g key={s.id}>
              <path d={d} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
              {showDots
                ? s.points.map((p, i) => (
                    <circle
                      key={i}
                      cx={xAt(i, n, w, pad.l, pad.r)}
                      cy={yAt(p.y, h, pad.t, pad.b)}
                      r="2.5"
                      fill={s.color}
                    />
                  ))
                : null}
            </g>
          );
        })}
        {hit && activeSeries && activePoint ? (
          <circle
            cx={hit.x}
            cy={hit.y}
            r="5"
            fill={activeSeries.color}
            className="spark-active"
          />
        ) : null}
        {xStart ? (
          <text x={pad.l} y={h - 6} className="spark-tick">
            {xStart}
          </text>
        ) : null}
        {xEnd ? (
          <text x={w - pad.r} y={h - 6} textAnchor="end" className="spark-tick">
            {xEnd}
          </text>
        ) : null}
      </svg>
      {hit && activeSeries && activePoint ? (
        <ChartTip x={hit.x} y={hit.y} vbW={w} vbH={h} pinned={pinned != null}>
          {renderTip({
            i: hit.pi,
            y: activePoint.y,
            color: activeSeries.color,
            series: activeSeries,
            data: activePoint.data,
            pinned: pinned != null,
          })}
        </ChartTip>
      ) : null}
      <div className="sr-only" aria-live="polite">
        {hit && activeSeries && activePoint && liveText
          ? liveText({ i: hit.pi, series: activeSeries, data: activePoint.data })
          : hit && activeSeries && activePoint
            ? `${activeSeries.id} point ${hit.pi + 1} value ${activePoint.y.toFixed(3)}`
            : ""}
      </div>
    </div>
  );
}

export type BarItem<T> = {
  y: number;
  color: string;
  label: string;
  data: T;
};

export function NativeBars<T>({
  bars,
  ariaLabel,
  floor,
  liveText,
  renderTip,
}: {
  bars: BarItem<T>[];
  ariaLabel: string;
  floor?: number;
  liveText?: (bar: BarItem<T>) => string;
  renderTip: (bar: BarItem<T>, ctx: { pinned: boolean }) => ReactNode;
}) {
  const w = CHART.w;
  const h = 200;
  const pad = { l: 36, r: 12, t: 12, b: 40 };
  const n = Math.max(bars.length, 1);
  const slot = (w - pad.l - pad.r) / n;
  const barW = Math.min(48, slot * 0.58);
  const layout = bars.map((bar, i) => {
    const x = pad.l + i * slot + (slot - barW) / 2;
    const y = yAt(bar.y, h, pad.t, pad.b);
    const height = h - pad.b - y;
    return { ...bar, i, x, y, height, cx: x + barW / 2 };
  });

  const { wrapRef, pinned, active, move, leave, toggle, keyNav } = usePinnedIndex(layout.length);
  const hit = active != null ? layout[active] : undefined;

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (layout.length === 0) return;
    const pt = svgCoords(event.currentTarget, event.clientX, event.clientY, w, h);
    const hits = layout.map((b) => ({ x: b.cx, y: b.y }));
    move(nearestIndex(hits, pt.x, pt.y));
  };

  return (
    <div className="chart-wrap" ref={wrapRef} onPointerLeave={leave}>
      <svg
        className="spark spark-bars"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onPointerMove={onMove}
        onPointerDown={(event) =>
          pinFromPointer(
            event,
            w,
            h,
            (pt) => (layout.length ? nearestIndex(layout.map((b) => ({ x: b.cx, y: b.y })), pt.x, pt.y) : -1),
            move,
            toggle,
          )
        }
        onKeyDown={keyNav}
      >
        <Grid w={w} h={h} pad={pad} ticks={Y_TICKS} floor={floor} />
        {layout.map((bar, i) => (
          <g key={bar.label}>
            <rect
              x={bar.x}
              y={bar.y}
              width={barW}
              height={Math.max(bar.height, 1)}
              rx="3"
              fill={bar.color}
              opacity={hit && hit.i !== i ? 0.45 : 1}
            />
            {bars.length <= BAR_LIMIT ? (
              <circle cx={bar.cx} cy={bar.y} r={hit?.i === i ? 4.5 : 2.5} fill={bar.color} className={hit?.i === i ? "spark-active" : undefined} />
            ) : null}
            <text x={bar.cx} y={h - 14} textAnchor="middle" className="spark-tick bar-label">
              {bar.label}
            </text>
          </g>
        ))}
      </svg>
      {hit ? (
        <ChartTip x={hit.cx} y={hit.y} vbW={w} vbH={h} pinned={pinned != null}>
          {renderTip(hit, { pinned: pinned != null })}
        </ChartTip>
      ) : null}
      <div className="sr-only" aria-live="polite">
        {hit ? (liveText ? liveText(hit) : `${hit.label} ${hit.y.toFixed(3)}`) : ""}
      </div>
    </div>
  );
}

export type BarGroup<T> = {
  key: string;
  label: string;
  data: T;
  bars: { key: string; y: number; color: string }[];
};

export function NativeGroups<T>({
  groups,
  ariaLabel,
  liveText,
  renderTip,
}: {
  groups: BarGroup<T>[];
  ariaLabel: string;
  liveText?: (group: BarGroup<T>) => string;
  renderTip: (group: BarGroup<T>, ctx: { pinned: boolean }) => ReactNode;
}) {
  const w = CHART.w;
  const h = 200;
  const pad = { l: 36, r: 12, t: 12, b: 40 };
  const n = Math.max(groups.length, 1);
  const slot = (w - pad.l - pad.r) / n;
  const cluster = slot * 0.72;
  const barN = Math.max(groups[0]?.bars.length ?? 1, 1);
  const barW = cluster / barN;
  const layout = groups.map((group, gi) => {
    const origin = pad.l + gi * slot + (slot - cluster) / 2;
    const rects = group.bars.map((bar, bi) => {
      const x = origin + bi * barW;
      const y = yAt(bar.y, h, pad.t, pad.b);
      return { ...bar, x, y, height: h - pad.b - y };
    });
    const cx = origin + cluster / 2;
    const top = Math.min(...rects.map((r) => r.y));
    return { ...group, gi, origin, rects, cx, top, cluster };
  });

  const { wrapRef, pinned, active, move, leave, toggle, keyNav } = usePinnedIndex(layout.length);
  const hit = active != null ? layout[active] : undefined;

  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    if (layout.length === 0) return;
    const pt = svgCoords(event.currentTarget, event.clientX, event.clientY, w, h);
    const i = Math.max(0, Math.min(layout.length - 1, Math.floor((pt.x - pad.l) / slot)));
    move(i);
  };

  return (
    <div className="chart-wrap" ref={wrapRef} onPointerLeave={leave}>
      <svg
        className="spark spark-bars"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onPointerMove={onMove}
        onPointerDown={(event) =>
          pinFromPointer(
            event,
            w,
            h,
            (pt) => {
              if (layout.length === 0) return -1;
              return Math.max(0, Math.min(layout.length - 1, Math.floor((pt.x - pad.l) / slot)));
            },
            move,
            toggle,
          )
        }
        onKeyDown={keyNav}
      >
        <Grid w={w} h={h} pad={pad} ticks={Y_TICKS} />
        {layout.map((group) => (
          <g key={group.key}>
            {hit?.gi === group.gi ? (
              <rect
                x={group.origin - 4}
                y={pad.t}
                width={group.cluster + 8}
                height={h - pad.t - pad.b}
                fill="rgba(245, 165, 36, 0.08)"
                rx="4"
              />
            ) : null}
            {group.rects.map((bar) => (
              <rect
                key={bar.key}
                x={bar.x + 1}
                y={bar.y}
                width={Math.max(barW - 2, 2)}
                height={Math.max(bar.height, 1)}
                rx="2"
                fill={bar.color}
              />
            ))}
            <text x={group.cx} y={h - 14} textAnchor="middle" className="spark-tick bar-label">
              {group.label}
            </text>
          </g>
        ))}
      </svg>
      {hit ? (
        <ChartTip x={hit.cx} y={hit.top} vbW={w} vbH={h} pinned={pinned != null}>
          {renderTip(hit, { pinned: pinned != null })}
        </ChartTip>
      ) : null}
      <div className="sr-only" aria-live="polite">
        {hit ? (liveText ? liveText(hit) : hit.label) : ""}
      </div>
    </div>
  );
}
