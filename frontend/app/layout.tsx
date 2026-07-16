import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Document Chat — with Citations",
  description:
    "Ask questions about your documents and get answers where every claim cites its exact source (file + page).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
