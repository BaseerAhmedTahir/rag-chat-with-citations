"use client";

import { useState } from "react";
import type { Citation } from "@/lib/api";

// Splits answer text on [n] markers and renders each as a clickable chip that
// reveals the grounding passage. This is the core "verifiable citation" UX.
export function AnswerView({
  answer,
  citations,
}: {
  answer: string;
  citations: Citation[];
}) {
  const [active, setActive] = useState<number | null>(null);
  const byMarker = new Map(citations.map((c) => [c.marker, c]));

  const parts = answer.split(/(\[\d+\])/g);

  return (
    <div className="answer">
      <p className="answer-text">
        {parts.map((part, i) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (match) {
            const marker = Number(match[1]);
            const known = byMarker.has(marker);
            return (
              <button
                key={i}
                className={`marker${active === marker ? " active" : ""}`}
                disabled={!known}
                onClick={() => setActive(active === marker ? null : marker)}
                aria-label={`Show source ${marker}`}
              >
                {marker}
              </button>
            );
          }
          return <span key={i}>{part}</span>;
        })}
      </p>

      {citations.length > 0 && (
        <div className="citations">
          <h3>Sources</h3>
          {citations.map((c) => (
            <div
              key={c.marker}
              className={`citation${active === c.marker ? " active" : ""}`}
              id={`citation-${c.marker}`}
            >
              <div className="citation-head">
                <span className="citation-badge">{c.marker}</span>
                <span className="citation-source">
                  {c.source_file} — page {c.page_number}
                </span>
              </div>
              <p className="citation-snippet">{c.snippet}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
