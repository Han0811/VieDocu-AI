'use client';

import React, { useState } from 'react';
import { Copy, Check, FileText } from 'lucide-react';

interface OCRResultViewerProps {
  text: string;
}

export default function OCRResultViewer({ text }: OCRResultViewerProps) {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-panel p-5 rounded-2xl flex flex-col h-full space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <FileText className="h-4.5 w-4.5 text-violet-400" />
          Kết quả Văn bản (Full Text)
        </h3>
        
        <button
          onClick={handleCopy}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-slate-700 bg-slate-800/50 hover:bg-slate-800 text-slate-300 hover:text-white flex items-center gap-1.5 transition-all"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-emerald-400" />
              Đã sao chép
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              Sao chép văn bản
            </>
          )}
        </button>
      </div>

      <div className="flex-1 bg-slate-950/80 rounded-xl border border-slate-800/80 p-4 font-mono text-sm leading-relaxed text-slate-100 overflow-y-auto max-h-[500px] whitespace-pre-wrap select-text selection:bg-violet-500/30 selection:text-white">
        {text ? text : <span className="text-slate-500 italic">Không có văn bản nhận diện được hoặc trống.</span>}
      </div>
    </div>
  );
}
