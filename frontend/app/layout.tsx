import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Drug Repurposing BioAgent",
  description: "AI-driven biomedical reasoning with FastAPI, LangGraph, and STRING",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
