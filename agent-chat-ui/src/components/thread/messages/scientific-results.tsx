import type { ReactNode } from "react";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const paper = z.object({
  title: z.string(),
  identifiers: z.record(z.string(), z.string().nullable()).optional(),
  authors: z
    .array(z.union([z.string(), z.object({ name: z.string() })]))
    .optional(),
  year: z.number().nullish(),
  url: z.string().nullish(),
  sources: z
    .record(z.string(), z.object({ url: z.string().nullish() }))
    .optional(),
});
const evidence = z.object({
  id: z.string(),
  paper,
  passage: z.string(),
  page: z.number().int().positive().nullish(),
  pdf_url: z.string().nullish(),
  provenance: z.array(z.object({ source: z.string() })),
});
const verdicts = {
  supported: "Supported",
  contradicted: "Contradicted",
  insufficient_evidence: "Insufficient evidence",
};
const relations = {
  supports: "Supporting evidence",
  contradicts: "Contradicting evidence",
  context: "Context",
};
const executionStatuses = z.enum(["completed", "timeout", "unavailable"]);
const failureLabels = {
  timeout: "Verification timed out",
  unavailable: "Verification unavailable",
};

const report = z.object({
  status: executionStatuses.default("completed"),
  results: z.array(
    z.object({
      claim: z.object({ statement: z.string() }),
      verdict: z
        .enum(["supported", "contradicted", "insufficient_evidence"])
        .nullable(),
      status: executionStatuses.default("completed"),
      notices: z.array(z.string()).default([]),
      explanation: z.string(),
      citations: z.array(
        z.object({
          evidence,
          relation: z.enum(["supports", "contradicts", "context"]),
          quote: z.string().nullish(),
        }),
      ),
    }),
  ),
  notices: z.array(z.string()),
});
const corpus = z.object({
  evidence: z.array(evidence),
  notices: z.array(z.string()),
});
const research = z
  .object({
    papers: z.array(z.union([paper, z.object({ paper })])).optional(),
    paper: paper.nullish(),
    issues: z
      .array(z.object({ source: z.string(), code: z.string() }))
      .optional(),
  })
  .refine((value) => value.papers !== undefined || value.paper !== undefined);

function webUrl(value: string | null | undefined) {
  if (!value) return;
  try {
    const url = new URL(value);
    if (
      ["http:", "https:"].includes(url.protocol) &&
      !url.username &&
      !url.password
    ) {
      return url.href;
    }
  } catch {
    return;
  }
}

function PaperCitation({ reference }: { reference: z.infer<typeof paper> }) {
  const { arxiv, doi, openalex, semanticscholar, scifact } =
    reference.identifiers ?? {};
  const identifierLinks = [
    [doi, "https://doi.org/"],
    [arxiv, "https://arxiv.org/abs/"],
    [openalex, "https://openalex.org/"],
    [semanticscholar, "https://www.semanticscholar.org/paper/"],
  ];
  const href =
    webUrl(reference.url) ??
    identifierLinks
      .flatMap(([id, base]) =>
        id
          ? [webUrl(base + id.split("/").map(encodeURIComponent).join("/"))]
          : [],
      )
      .find(Boolean) ??
    Object.values(reference.sources ?? {})
      .map((source) => webUrl(source.url))
      .find(Boolean);
  const names = reference.authors
    ?.slice(0, 3)
    .map((author) => (typeof author === "string" ? author : author.name));
  const authors = names?.length
    ? names.join(", ") +
      ((reference.authors?.length ?? 0) > names.length ? " et al." : "")
    : null;
  const details = [
    authors,
    reference.year,
    scifact && `SciFact ${scifact}`,
  ].filter(Boolean);
  return (
    <div className="flex flex-col gap-1">
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="font-medium underline underline-offset-2"
        >
          {reference.title}
        </a>
      ) : (
        <span className="font-medium">{reference.title}</span>
      )}
      {details.length > 0 && (
        <p className="text-muted-foreground text-xs">{details.join(" · ")}</p>
      )}
    </div>
  );
}

function EvidencePassage({
  entry,
  relation,
  quote,
}: {
  entry: z.infer<typeof evidence>;
  relation?: keyof typeof relations;
  quote?: string | null;
}) {
  const providers = [
    ...new Set(
      entry.provenance.map(({ source }) =>
        source === "scifact" ? "SciFact" : "Research MCP",
      ),
    ),
  ];
  const pdfUrl = webUrl(entry.pdf_url);
  const pdfPage = pdfUrl && entry.page ? new URL(pdfUrl) : undefined;
  if (pdfPage) pdfPage.hash = `page=${entry.page}`;
  return (
    <details className="min-w-0">
      <summary className="cursor-pointer text-sm font-medium">
        {entry.id} · {relation ? relations[relation] : "Passage"}
      </summary>
      <div className="flex flex-col gap-2 py-2 pl-4 text-sm break-words">
        <PaperCitation reference={entry.paper} />
        <blockquote className="text-muted-foreground border-l pl-3">
          {quote ?? entry.passage}
        </blockquote>
        {pdfPage && (
          <a
            href={pdfPage.href}
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2"
          >
            PDF page {entry.page}
          </a>
        )}
        <p className="text-muted-foreground text-xs">{providers.join(" · ")}</p>
      </div>
    </details>
  );
}

function Notices({ entries }: { entries: string[] }) {
  if (!entries.length) return null;
  return (
    <ul className="text-muted-foreground list-disc pl-4 text-sm">
      {entries.map((message, index) => (
        <li key={index}>{message}</li>
      ))}
    </ul>
  );
}

export function ScientificResult({
  name,
  payload,
  fallback,
  onRetryClaim,
}: {
  name: string | undefined;
  payload: unknown;
  fallback: ReactNode;
  onRetryClaim?: (claim: string) => void;
}) {
  if (name === "fact_check") {
    const parsed = report.safeParse(payload);
    if (!parsed.success) return fallback;
    return (
      <section
        className="flex flex-col gap-3"
        aria-label="Fact-check report"
      >
        {parsed.data.status !== "completed" && (
          <p
            role="status"
            className="text-sm font-medium"
          >
            {failureLabels[parsed.data.status]}
          </p>
        )}
        {parsed.data.results.map((result, index) => (
          <Card key={index}>
            <CardHeader>
              <CardTitle
                role="heading"
                aria-level={3}
              >
                {result.status !== "completed"
                  ? failureLabels[result.status]
                  : result.verdict && verdicts[result.verdict]}
              </CardTitle>
              <CardDescription>{result.claim.statement}</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 break-words">
              <p className="text-sm">{result.explanation}</p>
              {result.citations.map(({ evidence: entry, relation, quote }) => (
                <EvidencePassage
                  key={entry.id}
                  entry={entry}
                  relation={relation}
                  quote={quote}
                />
              ))}
              <Notices entries={result.notices} />
              {result.status !== "completed" && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="self-start"
                  disabled={!onRetryClaim}
                  onClick={() => onRetryClaim?.(result.claim.statement)}
                >
                  Retry this claim
                </Button>
              )}
            </CardContent>
          </Card>
        ))}
        <Notices entries={parsed.data.notices} />
      </section>
    );
  }

  if (name === "retrieve_scifact") {
    const parsed = corpus.safeParse(payload);
    if (!parsed.success) return fallback;
    return (
      <Card>
        <CardHeader>
          <CardTitle>SciFact evidence</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {parsed.data.evidence.map((entry) => (
            <EvidencePassage
              key={entry.id}
              entry={entry}
            />
          ))}
          {parsed.data.evidence.length === 0 && (
            <p className="text-sm">No matching passages.</p>
          )}
          <Notices entries={parsed.data.notices} />
        </CardContent>
      </Card>
    );
  }

  const parsed = research.safeParse(payload);
  if (!parsed.success) return fallback;
  const papers =
    parsed.data.papers ?? (parsed.data.paper ? [parsed.data.paper] : []);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Research papers</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 break-words">
        {papers.length ? (
          <ol className="flex list-decimal flex-col gap-3 pl-4 text-sm">
            {papers.map((item, index) => (
              <li key={index}>
                <PaperCitation
                  reference={"paper" in item ? item.paper : item}
                />
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-sm">No papers returned.</p>
        )}
        <Notices
          entries={(parsed.data.issues ?? []).map(
            (issue) =>
              `Research provider ${issue.source} reported ${issue.code}.`,
          )}
        />
      </CardContent>
    </Card>
  );
}
