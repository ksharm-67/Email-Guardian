import { createFileRoute } from "@tanstack/react-router";
import { EmailClassifier } from "@/components/EmailClassifier";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Email triage — scam, spam, telemarketer, safe" },
      { name: "description", content: "Paste an email and get a TF-IDF + Naive Bayes classification with confidence and the tokens that drove the prediction." },
      { property: "og:title", content: "Email triage — scam, spam, telemarketer, safe" },
      { property: "og:description", content: "Paste an email and get a TF-IDF + Naive Bayes classification with confidence and the tokens that drove the prediction." },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <main className="min-h-screen bg-background">
      <EmailClassifier />
    </main>
  );
}
