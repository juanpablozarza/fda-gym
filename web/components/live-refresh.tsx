"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

/** Re-read public/results on an interval so a live eval shows up without a full reload. */
export function LiveRefresh({ seconds = 60 }: { seconds?: number }) {
  const router = useRouter();
  useEffect(() => {
    const id = window.setInterval(() => router.refresh(), seconds * 1000);
    return () => window.clearInterval(id);
  }, [router, seconds]);
  return null;
}
