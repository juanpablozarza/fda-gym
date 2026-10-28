"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  createColumnHelper,
  createPaginatedRowModel,
  createSortedRowModel,
  rowPaginationFeature,
  rowSortingFeature,
  tableFeatures,
  useTable,
} from "@tanstack/react-table";

import { fmtReward, kindLabel, replayPath, TRAJECTORY_PAGE_SIZE } from "@/lib/format";
import type { ModelKind } from "@/lib/types";

export type TrajectoryListRow = {
  id: string;
  taskId: string;
  question: string;
  modelName: string;
  kind: ModelKind;
  total: number;
  citationsValid: number;
  citationsTotal: number;
  tools: number;
};

const features = tableFeatures({
  rowSortingFeature,
  rowPaginationFeature,
  sortedRowModel: createSortedRowModel(),
  paginatedRowModel: createPaginatedRowModel(),
});

const helper = createColumnHelper<typeof features, TrajectoryListRow>();

const columns = helper.columns([
  helper.accessor("id", {
    header: "Id",
    enableSorting: false,
    cell: (info) => (
      <Link href={replayPath(info.getValue())} className="traj-id" onClick={(event) => event.stopPropagation()}>
        {info.getValue()}
      </Link>
    ),
  }),
  helper.accessor("taskId", {
    header: "Task",
    enableSorting: false,
    cell: (info) => (
      <>
        <div className="mono sm muted">{info.getValue()}</div>
        <div className="tq">{info.row.original.question}</div>
      </>
    ),
  }),
  helper.accessor("modelName", {
    header: "Model",
    enableSorting: false,
    cell: (info) => (
      <>
        <div className="mname">{info.getValue()}</div>
        <span className="chip">{kindLabel(info.row.original.kind)}</span>
      </>
    ),
  }),
  helper.accessor("total", {
    header: "Total",
    cell: (info) => <span className="mono b">{fmtReward(info.getValue())}</span>,
  }),
  helper.display({
    id: "cite",
    header: "Cite",
    enableSorting: false,
    cell: ({ row }) => (
      <span className="mono">
        {row.original.citationsValid}/{row.original.citationsTotal}
      </span>
    ),
  }),
  helper.accessor("tools", {
    header: "Tools",
    cell: (info) => <span className="mono">{info.getValue()}</span>,
  }),
]);

const NUMERIC = new Set(["total", "cite", "tools"]);

function headerClass(id: string, canSort: boolean): string | undefined {
  const bits = [NUMERIC.has(id) ? "num" : "", canSort ? "sortable" : ""].filter(Boolean);
  return bits.length ? bits.join(" ") : undefined;
}

export function TrajectoriesTable({
  rows,
  hidePager = false,
}: {
  rows: TrajectoryListRow[];
  hidePager?: boolean;
}) {
  const router = useRouter();
  const table = useTable({
    features,
    columns,
    data: rows,
    initialState: { pagination: { pageIndex: 0, pageSize: TRAJECTORY_PAGE_SIZE } },
  });

  const pageRows = table.getRowModel().rows;
  const total = table.getRowCount();

  return (
    <>
      <table>
        <thead>
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => {
                const canSort = header.column.getCanSort();
                const sorted = header.column.getIsSorted();
                return (
                  <th
                    key={header.id}
                    className={headerClass(header.column.id, canSort)}
                    onClick={canSort ? (event) => header.column.getToggleSortingHandler()?.(event) : undefined}
                  >
                    {header.isPlaceholder ? null : <table.FlexRender header={header} />}
                    {sorted === "asc" ? " ↑" : sorted === "desc" ? " ↓" : null}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>
        <tbody>
          {pageRows.map((row) => {
            const href = replayPath(row.original.id);
            return (
              <tr
                key={row.id}
                className="clickable"
                onClick={(event) => {
                  if ((event.target as HTMLElement).closest("a")) return;
                  if (event.metaKey || event.ctrlKey) {
                    window.open(href, "_blank", "noopener,noreferrer");
                    return;
                  }
                  router.push(href);
                }}
                onMouseEnter={() => router.prefetch(href)}
              >
                {row.getAllCells().map((cell) => (
                  <td
                    key={cell.id}
                    className={NUMERIC.has(cell.column.id) ? "num" : cell.column.id === "id" ? "mono" : undefined}
                  >
                    <table.FlexRender cell={cell} />
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {hidePager ? null : (
        <div className="pager">
          <span className="chip">{TRAJECTORY_PAGE_SIZE} / page</span>
          <span className="muted sm">
            {pageRows.length} of {total}
          </span>
          <button
            type="button"
            className="btn ghost"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            ◀ Prev
          </button>
          <button type="button" className="btn ghost" onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>
            Next ▶
          </button>
        </div>
      )}
    </>
  );
}
