import { useMemo, useState } from "react";
import { train, predict, LABELS, type TrainedModel } from "@/lib/classifier/classifier";
import type { Label } from "@/lib/classifier/examples";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

const SAMPLES: { name: string; text: string }[] = [
  {
    name: "Phishing",
    text: "URGENT: Your account has been suspended due to suspicious activity. Click the link to verify your identity and restore access within 24 hours: http://secure-verify.co/login",
  },
  {
    name: "Marketing blast",
    text: "VIAGRA 80% OFF!!! Cheap meds no prescription needed. Order now and get FREE shipping worldwide — limited time only.",
  },
  {
    name: "Cold call script",
    text: "Hi, this is Jessica from HomeShield Warranty. We're calling about extending coverage on your vehicle. Press 1 to speak with an agent.",
  },
  {
    name: "Coworker note",
    text: "Hey, the PR is ready for review. CI is green and I added unit tests for the new auth flow. Let me know if you want to pair on it tomorrow.",
  },
];

const LABEL_CLASSES: Record<Label, { bg: string; text: string; bar: string; ring: string }> = {
  scam:         { bg: "bg-label-scam",         text: "text-label-scam-foreground",         bar: "bg-label-scam",         ring: "ring-label-scam/30" },
  spam:         { bg: "bg-label-spam",         text: "text-label-spam-foreground",         bar: "bg-label-spam",         ring: "ring-label-spam/30" },
  telemarketer: { bg: "bg-label-telemarketer", text: "text-label-telemarketer-foreground", bar: "bg-label-telemarketer", ring: "ring-label-telemarketer/30" },
  safe:         { bg: "bg-label-safe",         text: "text-label-safe-foreground",         bar: "bg-label-safe",         ring: "ring-label-safe/30" },
};

export function EmailClassifier() {
  // Tiny dataset — training in a useMemo keeps it to one pass per session.
  const model: TrainedModel = useMemo(() => train(), []);
  const [text, setText] = useState("");
  const [result, setResult] = useState<ReturnType<typeof predict> | null>(null);

  const onClassify = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setResult(predict(model, trimmed));
  };

  const onSample = (s: string) => {
    setText(s);
    setResult(predict(model, s));
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-6 py-12 sm:py-16">
      <header className="mb-10">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
          TF-IDF · Multinomial Naive Bayes
        </p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
          Email triage
        </h1>
        <p className="mt-3 max-w-xl text-sm text-muted-foreground">
          Paste an email, voicemail transcript, or SMS and the model will classify it as
          <span className="text-foreground"> scam</span>,
          <span className="text-foreground"> spam</span>,
          <span className="text-foreground"> telemarketer</span>, or
          <span className="text-foreground"> safe</span>. Runs entirely in your browser.
        </p>
      </header>

      <section className="rounded-xl border border-border bg-card p-5 shadow-sm">
        <label htmlFor="email" className="sr-only">Email text</label>
        <Textarea
          id="email"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste the message here…"
          className="min-h-40 resize-y border-0 bg-transparent p-0 text-sm leading-relaxed shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
        />
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
          <div className="flex flex-wrap gap-2">
            {SAMPLES.map((s) => (
              <button
                key={s.name}
                type="button"
                onClick={() => onSample(s.text)}
                className="rounded-full border border-border bg-background px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground"
              >
                {s.name}
              </button>
            ))}
          </div>
          <Button onClick={onClassify} disabled={!text.trim()} className="px-5">
            Classify
          </Button>
        </div>
      </section>

      {result && (
        <section className="mt-8 space-y-6">
          <div className={`flex items-center justify-between rounded-xl p-5 ring-1 ${LABEL_CLASSES[result.label].bg} ${LABEL_CLASSES[result.label].text} ${LABEL_CLASSES[result.label].ring}`}>
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.18em] opacity-80">
                Prediction
              </p>
              <p className="mt-1 text-2xl font-semibold tracking-tight">
                {result.label}
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs font-medium uppercase tracking-[0.18em] opacity-80">
                Confidence
              </p>
              <p className="mt-1 font-mono text-2xl font-semibold tabular-nums">
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card p-5">
            <h2 className="text-sm font-semibold text-foreground">Class probabilities</h2>
            <ul className="mt-4 space-y-3">
              {LABELS.map((l) => {
                const p = result.probabilities[l];
                return (
                  <li key={l}>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className="font-medium capitalize text-foreground">{l}</span>
                      <span className="font-mono tabular-nums text-muted-foreground">
                        {(p * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className={`h-full ${LABEL_CLASSES[l].bar} transition-[width] duration-500`}
                        style={{ width: `${Math.max(p * 100, 1.5)}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>

          <div className="rounded-xl border border-border bg-card p-5">
            <div className="flex items-baseline justify-between">
              <h2 className="text-sm font-semibold text-foreground">
                Top indicative features
              </h2>
              <span className="text-xs text-muted-foreground">
                {result.vocabHits} tokens matched the model vocabulary
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Tokens that pushed the prediction toward
              <span className="ml-1 font-medium capitalize text-foreground">{result.label}</span>,
              ranked by their contribution to the log-score.
            </p>
            {result.topFeatures.length === 0 ? (
              <p className="mt-4 text-sm text-muted-foreground">
                No strongly indicative tokens — the prediction relies on the class prior.
              </p>
            ) : (
              <ul className="mt-4 flex flex-wrap gap-2">
                {result.topFeatures.map((f) => (
                  <li
                    key={f.token}
                    className="inline-flex items-center gap-2 rounded-md border border-border bg-background px-2.5 py-1 font-mono text-xs"
                  >
                    <span className="text-foreground">{f.token}</span>
                    <span className="tabular-nums text-muted-foreground">
                      {f.weight.toFixed(3)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
