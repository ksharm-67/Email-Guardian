# Email Triage Classifier

A browser-based email classifier that labels messages as **scam**, **spam**, **telemarketer**, or **safe** using TF-IDF and Multinomial Naive Bayes — entirely client-side.

## How it works

Paste an email, voicemail transcript, or SMS into the web app. The model tokenizes the text, computes TF-IDF scores, and runs a Multinomial Naive Bayes classifier trained on labeled examples. You get:

- **Predicted label** with confidence percentage
- **Per-class probability bars** so you can see how close the runner-up classes were
- **Top indicative features** — the specific tokens that pushed the prediction toward the winning class

All computation happens in the browser. No data is sent to a server.

## Tech stack

- [TanStack Start](https://tanstack.com/start) — full-stack React framework
- React 19 + TypeScript
- Tailwind CSS v4 + shadcn/ui components
- Custom TF-IDF + Naive Bayes implementation in TypeScript

## Model details

| Property | Value |
|----------|-------|
| Algorithm | Multinomial Naive Bayes |
| Features | TF-IDF unigrams & bigrams |
| Classes | scam, spam, telemarketer, safe |
| Training data | 60 hand-labeled examples (15 per class) |
| Runtime | Entirely in-browser, ~50 ms inference |

## Project structure

```
src/
  components/
    EmailClassifier.tsx   # Main UI component
  lib/classifier/
    classifier.ts         # train() and predict() implementation
    examples.ts           # Labeled training data
  routes/
    index.tsx             # Home page
    __root.tsx            # Root layout
```

## Development

```bash
# Install dependencies
bun install

# Start dev server
bun dev

# Build for production
bun run build
```

## Improving accuracy

The current model is trained on a small seed dataset. To improve real-world accuracy, add more labeled examples to `src/lib/classifier/examples.ts` and retrain in-browser automatically on page load.

## License

MIT
