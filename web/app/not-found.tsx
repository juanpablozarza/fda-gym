import Link from "next/link";

export default function NotFound() {
  return (
    <>
      <header className="top">
        <div>
          <h1>Not found</h1>
          <div className="muted">No trajectory or page at this URL.</div>
        </div>
      </header>
      <section className="card glass">
        <p className="muted">
          Check the id against <Link href="/runs">Runs</Link> or <Link href="/trajectories">Trajectories</Link>, or
          return to <Link href="/">Overview</Link>.
        </p>
      </section>
    </>
  );
}
