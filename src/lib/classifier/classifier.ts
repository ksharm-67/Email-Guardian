/**
 * Multinomial Naive Bayes over TF-IDF features (word unigrams + bigrams).
 * Ported from the Python sklearn pipeline; trains in-browser in < 50ms on the
 * seed dataset, so we don't need a server round-trip.
 */
import { EXAMPLES, type Label } from "./examples";

export const LABELS: Label[] = ["scam", "spam", "telemarketer", "safe"];

const STOPWORDS = new Set([
  "a","an","and","are","as","at","be","by","for","from","has","have","i","in",
  "is","it","its","of","on","or","that","the","to","was","were","will","with",
  "you","your","this","my","we","our","us","but","not","if","so","do","does",
]);

/** Lowercase, split on non-word, drop very short tokens and stopwords. */
function words(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9$]+/i)
    .filter((t) => t.length >= 2 && !STOPWORDS.has(t));
}

/** Word unigrams + bigrams, mirroring sklearn ngram_range=(1,2). */
function tokenize(text: string): string[] {
  const w = words(text);
  const out: string[] = [...w];
  for (let i = 0; i < w.length - 1; i++) out.push(`${w[i]} ${w[i + 1]}`);
  return out;
}

/** Sparse term-frequency map for one document. */
function termFreq(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}

export interface TrainedModel {
  vocab: Map<string, number>;          // token -> feature index
  idf: Float64Array;                   // [V]
  classLogPrior: Record<Label, number>;
  featureLogProb: Record<Label, Float64Array>; // [V]
  trainedAt: number;
  vocabSize: number;
  numDocs: number;
}

/** L2-normalized sublinear TF-IDF vector for one document. */
function tfidfVector(
  tf: Map<string, number>,
  vocab: Map<string, number>,
  idf: Float64Array,
): Map<number, number> {
  const v = new Map<number, number>();
  let sq = 0;
  for (const [tok, count] of tf) {
    const idx = vocab.get(tok);
    if (idx === undefined) continue;
    const w = (1 + Math.log(count)) * idf[idx]; // sublinear_tf
    v.set(idx, w);
    sq += w * w;
  }
  const norm = Math.sqrt(sq);
  if (norm > 0) for (const [k, val] of v) v.set(k, val / norm);
  return v;
}

export function train(alpha = 0.3): TrainedModel {
  const docs = EXAMPLES.map((e) => ({ label: e.label, tokens: tokenize(e.text) }));
  const N = docs.length;

  // Build vocabulary + document frequency (min_df = 1).
  const df = new Map<string, number>();
  for (const d of docs) {
    const seen = new Set(d.tokens);
    for (const t of seen) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const vocab = new Map<string, number>();
  let i = 0;
  for (const tok of df.keys()) vocab.set(tok, i++);
  const V = vocab.size;

  // Smoothed IDF (sklearn default): log((1+N)/(1+df)) + 1.
  const idf = new Float64Array(V);
  for (const [tok, freq] of df) idf[vocab.get(tok)!] = Math.log((1 + N) / (1 + freq)) + 1;

  // Class priors + feature accumulators.
  const classCount: Record<Label, number> = { scam: 0, spam: 0, telemarketer: 0, safe: 0 };
  const featureCount: Record<Label, Float64Array> = {
    scam: new Float64Array(V),
    spam: new Float64Array(V),
    telemarketer: new Float64Array(V),
    safe: new Float64Array(V),
  };
  for (const d of docs) {
    classCount[d.label] += 1;
    const vec = tfidfVector(termFreq(d.tokens), vocab, idf);
    const acc = featureCount[d.label];
    for (const [idx, w] of vec) acc[idx] += w;
  }

  // log P(class) and smoothed log P(token | class).
  const classLogPrior = {} as Record<Label, number>;
  const featureLogProb = {} as Record<Label, Float64Array>;
  for (const c of LABELS) {
    classLogPrior[c] = Math.log(classCount[c] / N);
    const counts = featureCount[c];
    let total = 0;
    for (let k = 0; k < V; k++) total += counts[k];
    const denom = total + alpha * V;
    const lp = new Float64Array(V);
    for (let k = 0; k < V; k++) lp[k] = Math.log((counts[k] + alpha) / denom);
    featureLogProb[c] = lp;
  }

  return {
    vocab, idf, classLogPrior, featureLogProb,
    trainedAt: Date.now(), vocabSize: V, numDocs: N,
  };
}

export interface TokenContribution {
  token: string;
  weight: number; // contribution to the predicted class's log-score
}

export interface Prediction {
  label: Label;
  confidence: number;
  probabilities: Record<Label, number>;
  topFeatures: TokenContribution[];
  vocabHits: number;
}

/** Stable softmax over the per-class log-scores. */
function softmax(scores: Record<Label, number>): Record<Label, number> {
  const vals = LABELS.map((l) => scores[l]);
  const max = Math.max(...vals);
  const exps = vals.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const out = {} as Record<Label, number>;
  LABELS.forEach((l, i) => (out[l] = exps[i] / sum));
  return out;
}

export function predict(model: TrainedModel, text: string, topK = 6): Prediction {
  const tokens = tokenize(text);
  const tf = termFreq(tokens);
  const vec = tfidfVector(tf, model.vocab, model.idf);

  const scores = {} as Record<Label, number>;
  for (const c of LABELS) {
    let s = model.classLogPrior[c];
    const lp = model.featureLogProb[c];
    for (const [idx, w] of vec) s += w * lp[idx];
    scores[c] = s;
  }
  const probs = softmax(scores);

  let best: Label = LABELS[0];
  for (const c of LABELS) if (probs[c] > probs[best]) best = c;

  // Contribution per token = tfidf weight * (log P(t|best) - mean log P(t|others)).
  // Surfaces tokens that pushed the prediction toward `best` specifically.
  const others = LABELS.filter((l) => l !== best);
  const invVocab = new Map<number, string>();
  for (const [tok, idx] of model.vocab) invVocab.set(idx, tok);
  const contributions: TokenContribution[] = [];
  for (const [idx, w] of vec) {
    const lpBest = model.featureLogProb[best][idx];
    let mean = 0;
    for (const c of others) mean += model.featureLogProb[c][idx];
    mean /= others.length;
    const weight = w * (lpBest - mean);
    if (weight > 0) contributions.push({ token: invVocab.get(idx)!, weight });
  }
  contributions.sort((a, b) => b.weight - a.weight);

  return {
    label: best,
    confidence: probs[best],
    probabilities: probs,
    topFeatures: contributions.slice(0, topK),
    vocabHits: vec.size,
  };
}
