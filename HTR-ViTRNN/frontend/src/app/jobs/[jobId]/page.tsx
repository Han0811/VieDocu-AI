'use client';

import React, { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { getJobStatus, getJobResult, JobStatusResponse, OCRResultResponse } from '../../../lib/api';
import JobStatus from '../../../components/JobStatus';
import PageTabs from '../../../components/PageTabs';
import OCRResultViewer from '../../../components/OCRResultViewer';
import DebugImageViewer from '../../../components/DebugImageViewer';
import LineTable from '../../../components/LineTable';
import DownloadButtons from '../../../components/DownloadButtons';
import { ArrowLeft, RefreshCw, Zap, FileText } from 'lucide-react';

export default function JobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.jobId as string;

  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [result, setResult] = useState<OCRResultResponse | null>(null);
  const [selectedPageIndex, setSelectedPageIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Poll job status
  useEffect(() => {
    let timerId: NodeJS.Timeout | null = null;

    const checkStatus = async () => {
      try {
        const statusData = await getJobStatus(jobId);
        setStatus(statusData);
        setLoading(false);

        if (statusData.status === 'done' || statusData.status === 'failed') {
          if (timerId) clearInterval(timerId);
          if (statusData.status === 'done') {
            const resultData = await getJobResult(jobId);
            setResult(resultData);
          } else {
            setError(statusData.error || 'Job failed processing.');
          }
        }
      } catch (err: unknown) {
        console.error('Error polling job status:', err);
        setError('Không thể kết nối đến máy chủ API.');
        setLoading(false);
        if (timerId) clearInterval(timerId);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 1500);
    timerId = interval;

    return () => {
      clearInterval(interval);
    };
  }, [jobId]);

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center space-y-4">
        <RefreshCw className="h-8 w-8 text-violet-500 animate-spin" />
        <p className="text-slate-400 text-sm font-medium">Đang tải thông tin chi tiết Job...</p>
      </div>
    );
  }

  const selectedPage = result?.pages?.[selectedPageIndex];

  return (
    <main className="min-h-screen py-10 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto space-y-8">
      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => router.push('/')}
          className="flex items-center gap-2 text-slate-400 hover:text-white transition-all text-sm font-medium"
        >
          <ArrowLeft className="h-4 w-4" />
          Quay lại trang chủ
        </button>
        
        {status && (
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <span className="font-mono">ID: {jobId}</span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <Zap className="h-3.5 w-3.5 text-amber-500" /> Mode: {status.stage ? status.stage : 'Balanced'}
            </span>
          </div>
        )}
      </div>

      {/* Status Panel */}
      {status && (
        <JobStatus
          status={status.status}
          progress={status.progress}
          stage={status.stage}
          error={error || status.error}
        />
      )}

      {/* OCR Results Container */}
      {status?.status === 'done' && result && selectedPage && (
        <div className="space-y-6">
          {/* Header Actions / Downloads */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
            <div className="space-y-1">
              <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                <FileText className="h-6 w-6 text-violet-400" />
                Kết quả Nhận dạng OCR
              </h2>
              <p className="text-slate-400 text-xs sm:text-sm">
                Xử lý thành công {result.page_count} trang tài liệu.
              </p>
            </div>
            
            <DownloadButtons
              jobId={jobId}
              textUrl={selectedPage.files?.text}
              jsonUrl={selectedPage.files?.lines_json}
            />
          </div>

          {/* Page Selector Tabs */}
          <PageTabs
            pageCount={result.page_count}
            currentPageIndex={selectedPageIndex}
            onPageSelect={(idx) => setSelectedPageIndex(idx)}
          />

          {/* Main Visual Panels (Grid Side-by-Side) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <DebugImageViewer
              debugStyleRegions={selectedPage.files?.debug_style_regions}
              debugFinalLines={selectedPage.files?.debug_final_lines}
            />
            
            <OCRResultViewer text={selectedPage.text} />
          </div>

          {/* Detailed Lines Table */}
          <LineTable lines={selectedPage.lines || []} />
        </div>
      )}
    </main>
  );
}
