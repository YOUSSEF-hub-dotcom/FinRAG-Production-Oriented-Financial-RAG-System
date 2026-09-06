"use client"

import { Lightbulb } from "lucide-react"

interface SuggestedPromptsProps {
  tickerSymbol: string
  onSelect: (prompt: string) => void
}

const TICKER_PROMPTS: Record<string, string[]> = {
  AAPL: [
    "What are Apple's top risk factors in the 2025 10-K filing?",
    "Analyze Apple's revenue segment breakdown for FY2025",
    "How does Apple's Services segment margin compare to Products?",
    "Summarize Apple's R&D spending trends over the last 3 years",
    "What is Apple's capital return program status in 2025?",
    "Explain Apple's supply chain risks mentioned in the 10-K",
  ],
  MSFT: [
    "What are the key growth drivers for Microsoft Azure in 2025?",
    "Analyze Microsoft's AI capital expenditure plans",
    "Summarize Microsoft's revenue by reporting segment",
    "What regulatory risks does Microsoft disclose in the 10-K?",
    "Compare Microsoft's cloud vs on-premise revenue trends",
    "What is Microsoft's cash flow from operations trend?",
  ],
  NVDA: [
    "Analyze NVIDIA's data center revenue growth trajectory",
    "What are NVIDIA's competitive advantages in AI chips?",
    "Summarize NVIDIA's gross margin expansion drivers",
    "What export restriction risks does NVIDIA disclose?",
    "Compare NVIDIA's R&D intensity across fiscal years",
    "Explain NVIDIA's Blackwell architecture market opportunity",
  ],
}

const GENERAL_PROMPTS = [
  "Compare the financial health of AAPL, MSFT, and NVDA",
  "Which company has the best operating margin trend?",
  "Summarize key industry risks across all three tech giants",
]

export function SuggestedPrompts({ tickerSymbol, onSelect }: SuggestedPromptsProps) {
  const prompts = [...(TICKER_PROMPTS[tickerSymbol] || []), ...GENERAL_PROMPTS]

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Lightbulb className="h-4 w-4" />
        Suggested questions
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {prompts.map((prompt) => (
          <button
            key={prompt}
            onClick={() => onSelect(prompt)}
            className="rounded-lg border bg-card p-3 text-left text-xs text-card-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  )
}
