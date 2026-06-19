'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { submitJob, listJobs, deleteJob, JobItem } from '../lib/api';
import UploadPanel from '../components/UploadPanel';
import { History, Eye, Trash2, CheckCircle2, Loader2, AlertTriangle, FileText, Calendar } from 'lucide-react';

export default function Home() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = async () => {
    try {
      const data = await listJobs(15);
      setJobs(data);
    } catch (err) {
      console.error('Failed to fetch jobs:', err);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUpload = async (
    file: File,
    options: { mode: string; paddleDevice: string; qwenEnabled: boolean }
  ) => {
    setIsSubmitting(true);
    setError(null);
    try {
      const res = await submitJob(file, {
        mode: options.mode,
        paddleDevice: options.paddleDevice,
        qwenEnabled: options.qwenEnabled,
      });
      router.push(`/jobs/${res.job_id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg || 'Lỗi khi tải tập tin hoặc tạo job');
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (jobId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Bạn có chắc chắn muốn xoá job này và toàn bộ kết quả?')) return;
    try {
      await deleteJob(jobId);
      fetchJobs();
    } catch (err) {
      console.error('Failed to delete job:', err);
      alert('Không thể xoá job');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'done':
        return <CheckCircle2 className="h-4 w-4 text-emerald-400" />;
      case 'failed':
        return <AlertTriangle className="h-4 w-4 text-rose-400" />;
      case 'processing':
        return <Loader2 className="h-4 w-4 text-violet-400 animate-spin" />;
      default:
        return <Loader2 className="h-4 w-4 text-sky-400 animate-pulse-slow" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'done': return 'Hoàn thành';
      case 'failed': return 'Thất bại';
      case 'processing': return 'Đang xử lý';
      case 'queued': return 'Đang chờ';
      default: return status;
    }
  };

  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleString('vi-VN', {
        hour: '2-digit',
        minute: '2-digit',
        day: '2-digit',
        month: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <main className="min-h-screen py-12 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center space-y-3">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-violet-400 via-indigo-200 to-cyan-400 bg-clip-text text-transparent">
          Document OCR System
        </h1>
        <p className="text-slate-400 max-w-xl mx-auto text-sm md:text-base leading-relaxed">
          Nhận diện chữ viết tay và in ấn tiếng Việt hiệu năng cao sử dụng HTR-ViTRNN và hậu kiểm thông minh với Qwen-VL.
        </p>
      </div>

      {error && (
        <div className="max-w-2xl mx-auto border border-rose-500/20 bg-rose-500/10 text-rose-300 px-4 py-3 rounded-xl text-sm text-center">
          {error}
        </div>
      )}

      {/* Upload Component */}
      <UploadPanel onUpload={handleUpload} isSubmitting={isSubmitting} />

      {/* Job History */}
      <div className="glass-panel p-6 sm:p-8 rounded-2xl space-y-6">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <History className="h-5 w-5 text-violet-400" />
          Lịch sử các Job OCR
        </h2>

        {jobs.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-slate-800 rounded-xl bg-slate-950/10 text-slate-500 text-sm">
            Chưa có job nào được thực hiện. Hãy bắt đầu bằng cách tải lên một tập tin!
          </div>
        ) : (
          <div className="overflow-x-auto border border-slate-800/80 rounded-xl">
            <table className="w-full border-collapse text-left text-sm text-slate-300">
              <thead className="bg-slate-900/80 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="py-3 px-4">Tên tập tin</th>
                  <th className="py-3 px-4">Ngày tạo</th>
                  <th className="py-3 px-4 w-28">Chế độ</th>
                  <th className="py-3 px-4 w-32">Trạng thái</th>
                  <th className="py-3 px-4 w-24 text-center">Thao tác</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-950/20">
                {jobs.map((job) => (
                  <tr
                    key={job.job_id}
                    onClick={() => router.push(`/jobs/${job.job_id}`)}
                    className="hover:bg-slate-900/30 transition-colors cursor-pointer"
                  >
                    <td className="py-3.5 px-4 font-medium text-white flex items-center gap-2 max-w-xs sm:max-w-sm truncate">
                      <FileText className="h-4 w-4 text-slate-500 shrink-0" />
                      {job.original_filename}
                    </td>
                    <td className="py-3.5 px-4 text-slate-400 text-xs font-mono">
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {formatDate(job.created_at)}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold border border-slate-700 bg-slate-800 text-slate-300">
                        {job.mode}
                      </span>
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2 text-xs font-medium">
                        {getStatusIcon(job.status)}
                        <span>
                          {getStatusText(job.status)} 
                          {job.status === 'processing' && ` (${job.progress}%)`}
                        </span>
                      </div>
                    </td>
                    <td className="py-3.5 px-4">
                      <div className="flex items-center justify-center gap-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            router.push(`/jobs/${job.job_id}`);
                          }}
                          className="p-1.5 hover:bg-slate-850 rounded text-slate-400 hover:text-white transition"
                          title="Chi tiết"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          onClick={(e) => handleDelete(job.job_id, e)}
                          className="p-1.5 hover:bg-rose-500/10 rounded text-slate-400 hover:text-rose-400 transition"
                          title="Xoá"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
