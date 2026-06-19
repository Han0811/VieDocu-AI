'use client';

import React from 'react';
import { FileDown, FileJson, FolderArchive } from 'lucide-react';
import { getJobZipUrl, getApiUrl } from '../lib/api';

interface DownloadButtonsProps {
  jobId: string;
  textUrl?: string;
  jsonUrl?: string;
}

export default function DownloadButtons({ jobId, textUrl, jsonUrl }: DownloadButtonsProps) {
  const triggerDownload = (url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownloadText = async () => {
    if (!textUrl) return;
    triggerDownload(getApiUrl(textUrl), `ocr_${jobId}.txt`);
  };

  const handleDownloadJson = async () => {
    if (!jsonUrl) return;
    triggerDownload(getApiUrl(jsonUrl), `ocr_${jobId}_lines.json`);
  };

  const handleDownloadZip = () => {
    triggerDownload(getJobZipUrl(jobId), `ocr_${jobId}.zip`);
  };

  return (
    <div className="flex flex-wrap gap-3">
      {textUrl && (
        <button
          onClick={handleDownloadText}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-slate-700 bg-slate-800/40 hover:bg-slate-800 text-sm font-semibold text-slate-300 hover:text-white transition-all duration-200"
        >
          <FileDown className="h-4 w-4 text-violet-400" />
          Tải file TXT
        </button>
      )}

      {jsonUrl && (
        <button
          onClick={handleDownloadJson}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-slate-700 bg-slate-800/40 hover:bg-slate-800 text-sm font-semibold text-slate-300 hover:text-white transition-all duration-200"
        >
          <FileJson className="h-4 w-4 text-amber-400" />
          Tải JSON kết quả
        </button>
      )}

      <button
        onClick={handleDownloadZip}
        className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-violet-500/30 bg-violet-500/10 hover:bg-violet-500/20 text-sm font-semibold text-violet-300 hover:text-white transition-all duration-200"
      >
        <FolderArchive className="h-4 w-4" />
        Tải ZIP trọn bộ (All-in-one)
      </button>
    </div>
  );
}
