import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { Shell } from "@/components/shell";
import { loadShell } from "@/lib/results";

import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FDA-Gym",
  description: "Replay openFDA regulatory trajectories. Dark glass, signal amber.",
};

export const dynamic = "force-dynamic";

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const shell = loadShell();
  return (
    <html lang="en">
      <body className={`${inter.variable} ${jetbrains.variable} antialiased`}>
        <Shell env={shell.env} nTasks={shell.nTasks} nSources={shell.nSources} snapshot={shell.snapshot}>
          {children}
        </Shell>
      </body>
    </html>
  );
}
