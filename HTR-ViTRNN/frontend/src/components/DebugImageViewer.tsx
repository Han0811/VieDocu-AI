'use client';

import React, { useState } from 'react';
import { Eye, Image as ImageIcon, Sparkles } from 'lucide-react';
import { getApiUrl } from '../lib/api';

interface DebugImageViewerProps {
  debugStyleRegions: string;
  debugFinalLines: string;
}

export default function DebugImageViewer({ debugStyleRegions, debugFinalLines }: DebugImageViewerProps) {
  const [activeTab, setActiveTab] = useState<'style' | 'lines'>('style');

  const getActiveImageUrl = () => {
    const rawPath = activeTab === 'style' ? debugStyleRegions : debugFinalLines;
    return getApiUrl(rawPath);
  };

  return (
    <div className="glass-panel p-5 rounded-2xl flex flex-col h-full space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Eye className="h-4.5 w-4.5 text-violet-400" />
          Ảnh Debug Bounding Boxes
        </h3>
        
        {/* Tab Buttons */}
        <div className="flex bg-slate-900/80 rounded-lg p-0.5 border border-slate-800">
          <button
            onClick={() => setActiveTab('style')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'style'
                ? 'bg-violet-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            Vùng chữ (Style Regions)
          </button>
          <button
            onClick={() => setActiveTab('lines')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all ${
              activeTab === 'lines'
                ? 'bg-violet-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ImageIcon className="h-3.5 w-3.5" />
            Dòng chữ (Final Lines)
          </button>
        </div>
      </div>

      {/* Image Display */}
      <div className="flex-1 bg-slate-950/80 rounded-xl border border-slate-800/80 flex items-center justify-center p-3 overflow-hidden min-h-[300px] md:min-h-[450px]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={getActiveImageUrl()}
          alt={activeTab === 'style' ? 'Style Regions Debug' : 'Final Lines Debug'}
          className="max-h-[500px] object-contain rounded transition-all duration-300 hover:scale-[1.02] cursor-zoom-in"
        />
      </div>
      
      <p className="text-center text-xs text-slate-500 italic">
        * Hover/di chuột để zoom nhẹ. Xem phân vùng chữ viết tay (handwriting) vs in ấn (printed).
      </p>
    </div>
  );
}
