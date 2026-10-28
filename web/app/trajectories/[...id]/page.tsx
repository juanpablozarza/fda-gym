import { notFound } from "next/navigation";

import { Replay } from "@/components/replay";
import { Caption } from "@/components/shell";
import { getTask, getTrajectory, joinTrajectoryId, loadCatalog, neighbors } from "@/lib/results";

export default async function TrajectoryPage({ params }: { params: Promise<{ id: string[] }> }) {
  const { id } = await params;
  const trajectoryId = joinTrajectoryId(id);
  const trajectory = getTrajectory(trajectoryId);
  if (!trajectory) notFound();
  const task = getTask(trajectory.task_id);
  const { prev, next } = neighbors(trajectoryId);
  const { example } = loadCatalog();

  return (
    <>
      <Replay trajectory={trajectory} task={task} prevId={prev?.id} nextId={next?.id} />
      <Caption example={example} />
    </>
  );
}
