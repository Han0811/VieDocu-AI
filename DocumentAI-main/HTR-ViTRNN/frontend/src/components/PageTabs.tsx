'use client';

import React from 'react';

interface PageTabsProps {
  pageCount: number;
  currentPageIndex: number;
  onPageSelect: (index: number) => void;
}

export default function PageTabs({ pageCount, currentPageIndex, onPageSelect }: PageTabsProps) {
  if (pageCount <= 1) return null;

  return (
    <div className="w-full flex border-b border-slate-800 overflow-x-auto no-scrollbar scroll-smooth">
      <div className="flex space-x-1 p-1 bg-slate-900/60 rounded-xl mb-4 border border-slate-800">
        {Array.from({ length: pageCount }).map((_, idx) => (
          <button
            key={idx}
            onClick={() => onPageSelect(idx)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              currentPageIndex === idx
                ? 'bg-violet-600 text-white shadow-md shadow-violet-600/10'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            Trang {idx + 1}
          </button>
        ))}
      </div>
    </div>
  );
}
