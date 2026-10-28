"use client";

import Link from "next/link";

import { NativeBars, NativeGroups, TipRow } from "@/components/native-chart";
import { fmtPct, fmtReward, trajectoriesPath } from "@/lib/format";
import type { PlanA } from "@/lib/training";

const BASE = "#8a97aa";
const SFT = "#62c9a5";
const GRPO = "#f5a524";
const LUNA = "#7eb6ff";
const HAIKU = "#d4a5ff";

type OverallBar = {
  kind: "base" | "trained" | "frontier";
  label: string;
  mean: number;
  passRate: number | null;
  citation: number | null;
  n: number | null;
  url?: string;
  hub?: string;
};

const TYPE_SHORT: Record<string, string> = {
  approval: "approval",
  faers_top_reaction: "top rxn",
  faers_count: "faers n",
  boxed_warning: "boxed",
  composite: "composite",
  label_open: "label",
};

function frontierColor(name: string, i: number): string {
  const lower = name.toLowerCase();
  if (lower.includes("luna")) return LUNA;
  if (lower.includes("haiku")) return HAIKU;
  return i % 2 === 0 ? LUNA : HAIKU;
}

function shortFrontier(name: string): string {
  if (/luna/i.test(name)) return "Luna";
  if (/haiku/i.test(name)) return "Haiku";
  return name.split(" ")[0] ?? name;
}

function overallBars(plan: PlanA): { bars: OverallBar[]; colors: string[] } {
  const bars: OverallBar[] = [
    {
      kind: "base",
      label: "untrained",
      mean: plan.untrained_test_mean,
      passRate: null,
      citation: null,
      n: null,
    },
  ];
  const colors = [BASE];
  for (const ckpt of plan.checkpoints) {
    if (ckpt.test_mean == null) continue;
    bars.push({
      kind: "trained",
      label: ckpt.stage === "sft" ? "SFT" : "GRPO×150",
      mean: ckpt.test_mean,
      passRate: ckpt.pass_rate,
      citation: ckpt.citation,
      n: null,
      url: ckpt.url,
      hub: ckpt.hub,
    });
    colors.push(ckpt.stage === "grpo" ? GRPO : SFT);
  }
  plan.frontier.forEach((row, i) => {
    bars.push({
      kind: "frontier",
      label: shortFrontier(row.name),
      mean: row.test_mean,
      passRate: row.pass_rate,
      citation: row.citation,
      n: row.n,
      hub: row.hub,
    });
    colors.push(frontierColor(row.name, i));
  });
  return { bars, colors };
}

function Overall({ plan }: { plan: PlanA }) {
  const { bars, colors } = overallBars(plan);
  return (
    <NativeBars
      bars={bars.map((bar, i) => ({
        y: bar.mean,
        color: colors[i] ?? GRPO,
        label: bar.label,
        data: bar,
      }))}
      ariaLabel="Plan A TEST overall mean reward. Hover a bar for pass rate and citations."
      floor={plan.untrained_test_mean}
      liveText={(bar) => `${bar.data.label} mean ${bar.data.mean.toFixed(4)}`}
      renderTip={(bar, { pinned }) => {
        const row = bar.data;
        return (
          <>
            <div className="tip-title">{row.label}</div>
            <TipRow k="mean R" v={fmtReward(row.mean, 4)} />
            {row.passRate != null ? <TipRow k="pass" v={fmtPct(row.passRate)} /> : null}
            {row.citation != null ? <TipRow k="citation" v={fmtPct(row.citation)} /> : null}
            {row.n != null ? <TipRow k="n" v={row.n} /> : null}
            {row.kind === "trained" && row.url ? (
              <a className="tip-go" href={row.url} target="_blank" rel="noreferrer">
                Hugging Face →
              </a>
            ) : null}
            {row.kind === "frontier" && row.hub ? <div className="muted sm mono">{row.hub}</div> : null}
            {pinned ? <div className="muted sm">pinned · Esc to clear</div> : null}
          </>
        );
      }}
    />
  );
}

function ByType({ plan }: { plan: PlanA }) {
  return (
    <>
      <NativeGroups
        groups={plan.by_type.map((row) => ({
          key: row.type,
          label: TYPE_SHORT[row.type] ?? row.type,
          data: row,
          bars: [
            { key: "baseline", y: row.baseline, color: BASE },
            { key: "sft", y: row.sft, color: SFT },
            { key: "grpo", y: row.grpo, color: GRPO },
          ],
        }))}
        ariaLabel="Plan A TEST mean reward by question type. Click a group for those trajectories."
        liveText={(g) =>
          `${g.data.type} n=${g.data.n} base ${g.data.baseline.toFixed(3)} sft ${g.data.sft.toFixed(3)} grpo ${g.data.grpo.toFixed(3)}`
        }
        renderTip={(group, { pinned }) => (
          <>
            <div className="tip-title mono">{group.data.type}</div>
            <TipRow k="n tasks" v={group.data.n} />
            <TipRow k="baseline" v={fmtReward(group.data.baseline, 3)} />
            <TipRow k="sft" v={fmtReward(group.data.sft, 3)} />
            <TipRow k="grpo" v={fmtReward(group.data.grpo, 3)} />
            <Link className="tip-go" href={trajectoriesPath({ type: group.data.type })}>
              trajectories →
            </Link>
            {pinned ? <div className="muted sm">pinned · Esc to clear</div> : null}
          </>
        )}
      />
      <div className="legend sm" style={{ marginTop: 8 }}>
        <span>
          <i className="lg-line" style={{ borderTopColor: BASE }} />
          baseline
        </span>
        <span>
          <i className="lg-line" style={{ borderTopColor: SFT }} />
          sft
        </span>
        <span>
          <i className="lg-line" style={{ borderTopColor: GRPO }} />
          grpo
        </span>
      </div>
    </>
  );
}

export function PlanACharts({ plan }: { plan: PlanA }) {
  return (
    <section className="charts">
      <figure className="card glass">
        <div className="ch">
          <h2>TEST overall</h2>
          <span className="muted sm">untrained · SFT · GRPO × 150 · Plan A frontier</span>
        </div>
        <Overall plan={plan} />
      </figure>
      <figure className="card glass">
        <div className="ch">
          <h2>TEST by type</h2>
          <span className="muted sm">strict score · click a group</span>
        </div>
        <ByType plan={plan} />
      </figure>
    </section>
  );
}
