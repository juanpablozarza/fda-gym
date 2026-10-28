import { fmtPct, fmtReward, fmtSigned } from "@/lib/format";
import type { PlanA } from "@/lib/training";

export function TrainingPath({ plan }: { plan: PlanA }) {
  const winner = plan.checkpoints.reduce((acc, c) => {
    if (c.test_mean == null) return acc;
    if (acc == null || (acc.test_mean ?? -1) < c.test_mean) return c;
    return acc;
  }, plan.checkpoints[0] ?? null);

  return (
    <>
      <section className="kpis">
        <div className="kpi glass">
          <div className="muted sm">Untrained 3B TEST</div>
          <div className="big mono">{fmtReward(plan.untrained_test_mean, 4)}</div>
          <div className="sm muted">greedy Plan A packed docs · strict score</div>
        </div>
        <div className="kpi glass">
          <div className="muted sm">Best adapter TEST</div>
          <div className="big mono">{winner?.test_mean == null ? "—" : fmtReward(winner.test_mean, 4)}</div>
          <div className="sm muted">
            {winner?.pass_rate == null ? "no eval" : `${fmtPct(winner.pass_rate)} pass · ${winner.label}`}
          </div>
        </div>
        <div className="kpi glass">
          <div className="muted sm">Reward lift (3B TEST)</div>
          <div className="big mono">{plan.lift == null ? "—" : fmtSigned(plan.lift, 4)}</div>
          <div className="sm muted">
            {plan.lift == null
              ? `gate Δ ≥ ${plan.beats_margin}`
              : `0.0185 → ${fmtReward(winner?.test_mean ?? 0, 4)} · gate Δ ≥ ${plan.beats_margin}`}
          </div>
        </div>
        <div className="kpi glass">
          <div className="muted sm">GRPO v2 train</div>
          <div className="big mono">~{fmtReward(plan.first_grpo.train_mean_approx)}</div>
          <div className="sm muted">format floor · no SFT</div>
        </div>
      </section>
      <p className="muted sm" style={{ margin: "4px 0 12px" }}>
        {plan.first_grpo.note}
      </p>
      <div className="adapters">
        {plan.checkpoints.map((ckpt) => (
          <article key={ckpt.id} className={`card glass adapter${ckpt.id === winner?.id ? " adapter-best" : ""}`}>
            <div className="ch" style={{ marginBottom: 4 }}>
              <h2>{ckpt.label}</h2>
              <span className={`chip${ckpt.stage === "grpo" ? " acc" : ""}`}>{ckpt.stage}</span>
            </div>
            <div className="mono b">{ckpt.test_mean == null ? "TEST —" : fmtReward(ckpt.test_mean, 4)}</div>
            <div className="sm muted">
              {ckpt.pass_rate == null ? "no held-out eval" : `${fmtPct(ckpt.pass_rate)} pass`}
              {ckpt.citation != null ? ` · cite ${fmtReward(ckpt.citation)}` : ""}
            </div>
            <p className="sm muted">{ckpt.detail}</p>
            <a className="sm mono" href={ckpt.url} target="_blank" rel="noreferrer">
              {ckpt.hub}
            </a>
          </article>
        ))}
      </div>
      {plan.by_type.length > 0 ? (
        <section className="card glass" style={{ marginBottom: 14 }}>
          <div className="ch">
            <h2>TEST by type</h2>
            <span className="muted sm">strict mean reward · 324 questions</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>Type</th>
                <th className="num">n</th>
                <th className="num">Base</th>
                <th className="num">SFT</th>
                <th className="num">GRPO × 150</th>
              </tr>
            </thead>
            <tbody>
              {plan.by_type.map((row) => (
                <tr key={row.type}>
                  <td className="mono">{row.type}</td>
                  <td className="num mono">{row.n}</td>
                  <td className="num mono">{fmtReward(row.baseline, 3)}</td>
                  <td className="num mono">{fmtReward(row.sft, 3)}</td>
                  <td className="num mono b">{fmtReward(row.grpo, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}
    </>
  );
}
