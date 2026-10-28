function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlight(raw: string): string {
  const escaped = escapeHtml(raw);
  return escaped.replace(
    /("(?:\\.|[^"\\])*")(\s*:)?|\b(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|\b(true|false|null)\b/g,
    (match, str: string | undefined, colon: string | undefined, num: string | undefined, kw: string | undefined) => {
      if (str && colon) return `<span class="k">${str}</span>${colon}`;
      if (str) return `<span class="s">${str}</span>`;
      if (num) return `<span class="n">${num}</span>`;
      if (kw) return `<span class="n">${kw}</span>`;
      return match;
    },
  );
}

export function JsonBlock({ value }: { value: unknown }) {
  const raw = value === undefined ? "null" : JSON.stringify(value, null, 2);
  return <pre className="json mono" dangerouslySetInnerHTML={{ __html: highlight(raw) }} />;
}
