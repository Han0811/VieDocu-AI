'use client';

import React, { useState } from 'react';
import { Copy, Check, ChevronDown, ChevronUp, Image as ImageIcon } from 'lucide-react';
import { LineResult, getApiUrl } from '../lib/api';

interface LineTableProps {
  lines: LineResult[];
}

export default function LineTable({ lines }: LineTableProps) {
  const [copiedLineId, setCopiedLineId] = useState<number | null>(null);
  const [expandedLineId, setExpandedLineId] = useState<number | null>(null);

  const copyToClipboard = (text: string, id: number) => {
    navigator.clipboard.writeText(text);
    setCopiedLineId(id);
    setTimeout(() => setCopiedLineId(null), 2000);
  };

  const toggleExpand = (id: number) => {
    setExpandedLineId(expandedLineId === id ? null : id);
  };

  const getRegionBadgeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'printed':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'handwriting':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'stamp':
        return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
      case 'mixed':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
      default:
        return 'bg-slate-500/10 text-slate-400 border-slate-500/20';
    }
  };

  const getVietnameseRegionType = (type: string) => {
    switch (type.toLowerCase()) {
      case 'printed': return 'In ấn';
      case 'handwriting': return 'Viết tay';
      case 'stamp': return 'Mộc đỏ';
      case 'mixed': return 'Hỗn hợp';
      default: return type;
    }
  };

  return (
    <div className="glass-panel p-5 rounded-2xl space-y-4">
      <h3 className="text-base font-bold text-white">Chi tiết từng dòng nhận diện ({lines.length} dòng)</h3>
      
      <div className="overflow-x-auto border border-slate-800 rounded-xl">
        <table className="w-full border-collapse text-left text-sm text-slate-300">
          <thead className="bg-slate-900/80 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <tr>
              <th className="py-3 px-4 w-12 text-center">ID</th>
              <th className="py-3 px-4">Ảnh cắt (Crop)</th>
              <th className="py-3 px-4">Kết quả OCR</th>
              <th className="py-3 px-4 w-32">Phân loại</th>
              <th className="py-3 px-4 w-24 text-center">Thao tác</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-950/20">
            {lines.map((line) => {
              const isExpanded = expandedLineId === line.line_id;
              return (
                <React.Fragment key={line.line_id}>
                  <tr className="hover:bg-slate-900/30 transition-colors">
                    <td className="py-3 px-4 text-center font-semibold text-slate-500">{line.line_id}</td>
                    <td className="py-3 px-4">
                      {line.line_crop_path ? (
                        <div 
                          className="relative max-w-[120px] max-h-[40px] bg-slate-900 border border-slate-800 rounded p-1 cursor-pointer hover:border-violet-500/50 transition"
                          onClick={() => toggleExpand(line.line_id)}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={getApiUrl(line.line_crop_path)}
                            alt={`crop-${line.line_id}`}
                            className="max-h-[30px] w-auto mx-auto object-contain"
                          />
                        </div>
                      ) : (
                        <span className="text-slate-600 flex items-center gap-1"><ImageIcon className="h-4 w-4" /> N/A</span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono font-medium text-white max-w-md break-words">
                      {line.text}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1">
                        {line.region_types.map((type, idx) => (
                          <span
                            key={idx}
                            className={`px-2 py-0.5 rounded text-[11px] font-semibold border ${getRegionBadgeColor(type)}`}
                          >
                            {getVietnameseRegionType(type)}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center justify-center gap-2">
                        <button
                          onClick={() => copyToClipboard(line.text, line.line_id)}
                          className="p-1.5 hover:bg-slate-800 rounded transition text-slate-400 hover:text-white"
                          title="Copy text"
                        >
                          {copiedLineId === line.line_id ? (
                            <Check className="h-4 w-4 text-emerald-400" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </button>
                        
                        <button
                          onClick={() => toggleExpand(line.line_id)}
                          className="p-1.5 hover:bg-slate-800 rounded transition text-slate-400 hover:text-white"
                          title="Expand parts"
                        >
                          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </button>
                      </div>
                    </td>
                  </tr>
                  
                  {/* Expanded Section */}
                  {isExpanded && (
                    <tr className="bg-slate-900/20">
                      <td colSpan={5} className="py-4 px-8 border-t border-b border-slate-800/80">
                        <div className="space-y-4">
                          {/* Main line crop magnified */}
                          {line.line_crop_path && (
                            <div className="space-y-1">
                              <span className="text-xs font-semibold text-slate-400">Ảnh dòng đầy đủ:</span>
                              <div className="inline-block bg-slate-900 p-2 rounded-lg border border-slate-800 max-w-full overflow-x-auto">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                  src={getApiUrl(line.line_crop_path)}
                                  alt={`line-crop-full-${line.line_id}`}
                                  className="max-h-[80px] w-auto object-contain"
                                />
                              </div>
                            </div>
                          )}

                          {/* Parts/Segments if any */}
                          {line.parts && line.parts.length > 0 && (
                            <div className="space-y-2">
                              <span className="text-xs font-semibold text-slate-400">Phân rã bộ phận (Word/Character Parts):</span>
                              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                                {line.parts.map((part) => (
                                  <div key={part.part_id} className="border border-slate-800/80 bg-slate-950/40 p-3 rounded-lg flex flex-col space-y-2">
                                    <div className="flex items-center justify-between">
                                      <span className="text-[10px] font-bold text-slate-500 uppercase">Part {part.part_id}</span>
                                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${getRegionBadgeColor(part.region_type)}`}>
                                        {getVietnameseRegionType(part.region_type)}
                                      </span>
                                    </div>
                                    {part.crop_path && (
                                      <div className="bg-slate-900 p-1 rounded border border-slate-800/50 flex items-center justify-center h-10 overflow-hidden">
                                        {/* eslint-disable-next-line @next/next/no-img-element */}
                                        <img
                                          src={getApiUrl(part.crop_path)}
                                          alt={`part-${part.part_id}`}
                                          className="max-h-[32px] w-auto object-contain"
                                        />
                                      </div>
                                    )}
                                    <div className="font-mono text-xs text-white break-words">{part.text}</div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
