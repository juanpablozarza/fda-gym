import { NextResponse } from "next/server";

import { episodesForTask } from "@/lib/results";

export function GET(request: Request) {
  const task = new URL(request.url).searchParams.get("task");
  if (!task) return NextResponse.json({ error: "task required" }, { status: 400 });
  return NextResponse.json(episodesForTask(task));
}
