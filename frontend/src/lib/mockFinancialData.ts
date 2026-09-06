export interface KeyRatio {
  label: string
  value: string
  change: number
  suffix?: string
}

export interface YearlyPerformance {
  year: number
  revenue: number
  netIncome: number
  rdSpend: number
}

export interface CompanyData {
  ticker: string
  name: string
  sector: string
  industry: string
  marketCap: string
  ceo: string
  filingStatus: string
  lastFilingDate: string
  keyRatios: KeyRatio[]
  performance: YearlyPerformance[]
  highlights: {
    risks: string[]
    opportunities: string[]
  }
}

export const MOCK_COMPANIES: Record<string, CompanyData> = {
  AAPL: {
    ticker: "AAPL",
    name: "Apple Inc.",
    sector: "Technology",
    industry: "Consumer Electronics",
    marketCap: "$3.42T",
    ceo: "Tim Cook",
    filingStatus: "Indexed",
    lastFilingDate: "2025-01-24",
    keyRatios: [
      { label: "Revenue", value: "$394.3B", change: 4.8 },
      { label: "EPS (Diluted)", value: "$6.42", change: 5.2 },
      { label: "Net Profit Margin", value: "26.2%", change: 1.1 },
      { label: "Operating Cash Flow", value: "$118.3B", change: 3.7 },
    ],
    performance: [
      { year: 2023, revenue: 383.3, netIncome: 97.0, rdSpend: 29.9 },
      { year: 2024, revenue: 391.0, netIncome: 99.8, rdSpend: 31.4 },
      { year: 2025, revenue: 394.3, netIncome: 101.3, rdSpend: 33.0 },
    ],
    highlights: {
      risks: [
        "iPhone revenue concentration (~52% of total)",
        "China market regulatory pressures",
        "Services segment antitrust scrutiny",
      ],
      opportunities: [
        "Apple Intelligence AI ecosystem expansion",
        "Services revenue growing 16% YoY",
        "Wearables & Home category new product cycle",
      ],
    },
  },
  MSFT: {
    ticker: "MSFT",
    name: "Microsoft Corporation",
    sector: "Technology",
    industry: "Software - Infrastructure",
    marketCap: "$3.18T",
    ceo: "Satya Nadella",
    filingStatus: "Indexed",
    lastFilingDate: "2025-01-29",
    keyRatios: [
      { label: "Revenue", value: "$245.1B", change: 16.2 },
      { label: "EPS (Diluted)", value: "$13.20", change: 18.4 },
      { label: "Net Profit Margin", value: "35.7%", change: 2.3 },
      { label: "Operating Cash Flow", value: "$89.2B", change: 21.5 },
    ],
    performance: [
      { year: 2023, revenue: 211.9, netIncome: 72.4, rdSpend: 27.2 },
      { year: 2024, revenue: 219.2, netIncome: 80.2, rdSpend: 29.5 },
      { year: 2025, revenue: 245.1, netIncome: 87.5, rdSpend: 32.1 },
    ],
    highlights: {
      risks: [
        "Azure growth deceleration risk vs AWS",
        "AI capital expenditure uncertainty ($80B+ planned)",
        "Activision Blizzard integration costs",
      ],
      opportunities: [
        "Copilot AI monetization across Office 365",
        "Azure AI services 60%+ revenue growth",
        "GitHub Copilot enterprise adoption surge",
      ],
    },
  },
  NVDA: {
    ticker: "NVDA",
    name: "NVIDIA Corporation",
    sector: "Technology",
    industry: "Semiconductors",
    marketCap: "$2.85T",
    ceo: "Jensen Huang",
    filingStatus: "Indexed",
    lastFilingDate: "2025-02-26",
    keyRatios: [
      { label: "Revenue", value: "$113.3B", change: 114.2 },
      { label: "EPS (Diluted)", value: "$2.94", change: 147.1 },
      { label: "Net Profit Margin", value: "55.8%", change: 12.4 },
      { label: "Operating Cash Flow", value: "$64.1B", change: 96.8 },
    ],
    performance: [
      { year: 2023, revenue: 27.0, netIncome: 9.7, rdSpend: 7.3 },
      { year: 2024, revenue: 60.9, netIncome: 29.8, rdSpend: 8.7 },
      { year: 2025, revenue: 113.3, netIncome: 63.2, rdSpend: 12.9 },
    ],
    highlights: {
      risks: [
        "Data center revenue cyclicality",
        "China export restriction impact",
        "Competitive pressure from AMD MI300X",
      ],
      opportunities: [
        "Blackwell GPU architecture massive demand",
        "Sovereign AI infrastructure buildout",
        "Automotive & robotics AI edge computing",
      ],
    },
  },
}

export const TICKER_LIST = ["AAPL", "MSFT", "NVDA"] as const
