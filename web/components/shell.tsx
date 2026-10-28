"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV = [
  { href: "/", label: "Overview", icon: "◧" },
  { href: "/runs", label: "Runs", icon: "▤" },
  { href: "/trajectories", label: "Trajectories", icon: "↳" },
  { href: "/tasks", label: "Tasks", icon: "☰" },
  { href: "/training", label: "Training", icon: "∿" },
  { href: "/docs", label: "Docs", icon: "¶" },
] as const;

function activePath(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Shell({
  children,
  env,
  nTasks,
  nSources,
  snapshot,
}: {
  children: React.ReactNode;
  env: string;
  nTasks: number;
  nSources: number;
  snapshot: string;
}) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <div className="t-b">
      <div className="bg" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
      <button
        type="button"
        className={`scrim${open ? " show" : ""}`}
        aria-label="Close menu"
        onClick={() => setOpen(false)}
      />
      <div className="app">
        <aside className={`side glass${open ? " open" : ""}`}>
          <Link href="/" className="brand" onClick={() => setOpen(false)}>
            <span className="mark" aria-hidden="true" />
            FDA‑Gym
          </Link>
          <div className="env mono">env: {env}</div>
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav${activePath(pathname, item.href) ? " on" : ""}`}
              onClick={() => setOpen(false)}
            >
              <span className="ic" aria-hidden="true">
                {item.icon}
              </span>
              {item.label}
            </Link>
          ))}
          <div className="sidefoot">
            <div className="muted sm">Dataset</div>
            <div className="mono sm">
              {nTasks.toLocaleString("en-US")} tasks · {nSources} sources
            </div>
            <div className="muted sm" style={{ marginTop: 6 }}>
              openFDA snapshot
            </div>
            <div className="mono sm">{snapshot}</div>
          </div>
        </aside>
        <main>
          <button type="button" className="menu" aria-label="Open menu" onClick={() => setOpen(true)}>
            ☰
          </button>
          {children}
        </main>
      </div>
    </div>
  );
}

export function Caption({ example }: { example: boolean }) {
  if (!example) return null;
  return <div className="caption">Example data · fictional drugs, models and records</div>;
}
