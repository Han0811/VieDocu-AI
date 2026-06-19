'use client';

import React, { useState, useRef, DragEvent } from 'react';
import { Upload, FileText, Cpu, Zap, Settings2 } from 'lucide-react';

interface UploadPanelProps {
  onUpload: (file: File, options: { mode: string; paddleDevice: string; qwenEnabled: boolean }) => void;
  isSubmitting: boolean;
}

export default function UploadPanel({ onUpload, isSubmitting }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<string>('BALANCED');
  const [paddleDevice, setPaddleDevice] = useState<string>('gpu:0');
  const [qwenEnabled, setQwenEnabled] = useState<boolean>(false);
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    onUpload(file, { mode, paddleDevice, qwenEnabled });
  };

  return (
    <form onSubmit={handleSubmit} className="glass-panel p-8 rounded-2xl max-w-2xl w-full mx-auto space-y-6">
      <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
        <Upload className="h-6 w-6 text-violet-400" />
        Tải lên tài liệu OCR
      </h2>

      {/* Drag & Drop Area */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
          isDragActive 
            ? 'border-violet-500 bg-violet-500/10' 
            : file 
              ? 'border-emerald-500/50 bg-emerald-500/5' 
              : 'border-slate-700 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          accept=".jpg,.jpeg,.png,.webp,.tif,.tiff,.pdf"
          onChange={handleFileChange}
        />
        
        {file ? (
          <div className="flex flex-col items-center text-center space-y-2">
            <FileText className="h-12 w-12 text-emerald-400" />
            <p className="text-white font-medium">{file.name}</p>
            <p className="text-xs text-slate-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center space-y-3">
            <Upload className="h-12 w-12 text-slate-500 animate-bounce" />
            <div className="text-sm">
              <span className="text-violet-400 font-semibold">Nhấn để chọn</span> hoặc kéo thả tập tin tại đây
            </div>
            <p className="text-xs text-slate-500 uppercase tracking-wider">
              PDF, JPG, PNG, WEBP, TIFF (Tối đa 100MB)
            </p>
          </div>
        )}
      </div>

      {/* Mode Selector */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-slate-300 flex items-center gap-1.5">
          <Zap className="h-4 w-4 text-amber-400" />
          Chế độ nhận diện (OCR Mode)
        </label>
        <div className="grid grid-cols-3 gap-3">
          {['FAST', 'BALANCED', 'ACCURATE'].map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`py-3 px-4 rounded-lg font-medium text-sm transition-all duration-200 border ${
                mode === m
                  ? 'bg-violet-600 text-white border-violet-500 shadow-lg shadow-violet-600/20'
                  : 'bg-slate-800/50 text-slate-300 border-slate-700 hover:bg-slate-800'
              }`}
            >
              {m === 'FAST' && 'Nhanh (Fast)'}
              {m === 'BALANCED' && 'Cân bằng'}
              {m === 'ACCURATE' && 'Chính xác'}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Paddle Device */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300 flex items-center gap-1.5">
            <Cpu className="h-4 w-4 text-sky-400" />
            Thiết bị xử lý (Paddle Device)
          </label>
          <select
            value={paddleDevice}
            onChange={(e) => setPaddleDevice(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg py-2.5 px-3 text-slate-300 text-sm focus:outline-none focus:border-violet-500"
          >
            <option value="gpu:0">GPU (RTX 5070 Ti)</option>
            <option value="cpu">CPU (Chậm hơn)</option>
          </select>
        </div>

        {/* Qwen VL Verification Toggle */}
        <div className="space-y-2 flex flex-col justify-end">
          <div className="flex items-center justify-between border border-slate-700 bg-slate-800/30 rounded-lg p-3">
            <div className="flex flex-col">
              <span className="text-sm font-medium text-slate-300 flex items-center gap-1.5">
                <Settings2 className="h-4 w-4 text-violet-400" />
                Dùng Qwen-VL sửa lỗi
              </span>
              <span className="text-xs text-slate-500">Mặc định tắt để tối ưu tốc độ</span>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={qwenEnabled}
                onChange={(e) => setQwenEnabled(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-300 after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-violet-600 peer-checked:after:bg-white"></div>
            </label>
          </div>
        </div>
      </div>

      <button
        type="submit"
        disabled={!file || isSubmitting}
        className="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-semibold py-4.5 rounded-xl transition-all duration-300 shadow-xl shadow-violet-600/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-base"
      >
        {isSubmitting ? (
          <>
            <span className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></span>
            Đang tải lên và khởi chạy job...
          </>
        ) : (
          'Bắt đầu nhận diện OCR'
        )}
      </button>
    </form>
  );
}
