import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Document OCR Production — HTR-ViTRNN",
  description: "High performance Vietnamese handwritten & printed text recognition system with Qwen-VL post-verification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="dark">
      <body className="antialiased selection:bg-violet-500/30 selection:text-violet-200">
        {children}
      </body>
    </html>
  );
}
