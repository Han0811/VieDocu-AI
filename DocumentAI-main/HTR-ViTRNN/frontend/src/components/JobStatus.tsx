'use client';

import React from 'react';
import { Loader2, CheckCircle2, AlertTriangle } from 'lucide-react';

interface JobStatusProps {
  status: string;
  progress: number;
  stage?: string;
  error?: string;
}

export default function JobStatus({ status, progress, stage, error }: JobStatusProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'queued':
        return 'text-sky-400 bg-sky-500/10 border-sky-500/30';
      case 'processing':
        return 'text-violet-400 bg-violet-500/10 border-violet-500/30';
      case 'done':
        return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
      case 'failed':
        return 'text-rose-400 bg-rose-500/10 border-rose-500/30';
      default:
        return 'text-slate-400 bg-slate-500/10 border-slate-500/30';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'queued':
        return 'Đang chờ xử lý';
      case 'processing':
        return 'Đang nhận diện';
      case 'done':
        return 'Đã hoàn thành';
      case 'failed':
        return 'Lỗi xử lý';
      default:
        return status;
    }
  };

  return (
    <div className="glass-panel p-6 rounded-2xl w-full mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Trạng thái tiến độ</span>
          {stage && <p className="text-white font-medium text-lg capitalize">{stage.replace(/_/g, ' ')}</p>}
        </div>
        <span className={`px-3 py-1.5 rounded-full text-xs font-semibold border flex items-center gap-1.5 ${getStatusColor()}`}>
          {status === 'processing' && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {status === 'queued' && <Loader2 className="h-3.5 w-3.5 animate-pulse-slow" />}
          {status === 'done' && <CheckCircle2 className="h-3.5 w-3.5" />}
          {status === 'failed' && <AlertTriangle className="h-3.5 w-3.5" />}
          {getStatusText()}
        </span>
      </div>

      {/* Progress Bar */}
      {status !== 'failed' && (
        <div className="space-y-2">
          <div className="w-full bg-slate-800 rounded-full h-3.5 overflow-hidden border border-slate-700/50">
            <div
              className="bg-gradient-to-r from-violet-500 to-indigo-500 h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-400">
            <span>Đang thực hiện...</span>
            <span className="font-semibold text-white">{progress}%</span>
          </div>
        </div>
      )}

      {/* Error Panel */}
      {status === 'failed' && error && (
        <div className="border border-rose-500/20 bg-rose-500/5 rounded-xl p-4 flex gap-3 items-start">
          <AlertTriangle className="h-5 w-5 text-rose-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-sm font-semibold text-white">Chi tiết lỗi</h4>
            <p className="text-sm text-rose-400/90 leading-relaxed font-mono whitespace-pre-wrap">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
