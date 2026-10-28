import { LiveRefresh } from "@/components/live-refresh";
import { PlanACharts } from "@/components/plan-a-charts";
import { Caption } from "@/components/shell";
import { TrainingCurve } from "@/components/training-curve";
import { TrainingPath } from "@/components/training-path";
import { loadCatalog } from "@/lib/results";
import { loadPlanA, loadTraining } from "@/lib/training";

export default function TrainingPage() {
  const { example } = loadCatalog();
  const runs = loadTraining();
  const plan = loadPlanA();
  return (
    <>
      <LiveRefresh />
      <header className="top">
        <div>
          <h1>Training</h1>
          <div className="muted">
            Plan A GRPO + LoRA on Qwen2.5-3B. Packed docs in the prompt; same strict score() as eval. Not the OpenRouter
            tool-calling table.
          </div>
        </div>
      </header>
      {plan ? <TrainingPath plan={plan} /> : null}
      <section className="card glass">
        <div className="ch">
          <h2>Training curve</h2>
          <span className="muted sm">mean dense reward per GRPO log step · dashed line is untrained TEST 0.0185</span>
        </div>
        <TrainingCurve runs={runs} />
      </section>
      {plan ? <PlanACharts plan={plan} /> : null}
      <Caption example={example} />
    </>
  );
}
