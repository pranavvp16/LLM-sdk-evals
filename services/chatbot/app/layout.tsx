import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ollive Chat",
  description: "Multi-provider LLM chatbot",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
