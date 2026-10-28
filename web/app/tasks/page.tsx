import { Caption } from "@/components/shell";
import { TaskExplorer } from "@/components/task-explorer";
import { loadCatalog } from "@/lib/results";

export default function TasksPage() {
  const { tasks, taskStats, example } = loadCatalog();
  return (
    <>
      <header className="top">
        <div>
          <h1>Tasks</h1>
          <div className="muted">Questions the fixture (and later the evals dump) can replay.</div>
        </div>
      </header>
      <TaskExplorer tasks={tasks} taskStats={taskStats} />
      <Caption example={example} />
    </>
  );
}
