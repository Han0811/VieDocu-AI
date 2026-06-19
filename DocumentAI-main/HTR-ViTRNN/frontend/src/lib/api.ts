export interface JobCreateResponse {
  job_id: string;
  status: string;
  message: string;
  created_at: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  progress: number;
  stage?: string;
  page_count?: number;
  current_page?: number;
  error?: string;
  result_url?: string;
  download_zip_url?: string;
  created_at: string;
  updated_at: string;
}

export interface PartResult {
  part_id: number;
  region_id: number;
  region_type: string;
  text: string;
  raw_text: string;
  crop_path: string;
  box_xyxy: [number, number, number, number];
  local_box_xyxy: [number, number, number, number];
  dropped: boolean;
}

export interface LineResult {
  line_id: number;
  text: string;
  raw_text: string;
  line_crop_path: string;
  box_xyxy: [number, number, number, number];
  region_types: string[];
  parts: PartResult[];
}

export interface PageResult {
  page_index: number;
  page_name: string;
  text: string;
  lines: LineResult[];
  files: {
    text: string;
    lines_json: string;
    regions_json: string;
    metadata: string;
    debug_style_regions: string;
    debug_final_lines: string;
  };
}

export interface OCRResultResponse {
  job_id: string;
  status: string;
  page_count: number;
  pages: PageResult[];
}

export interface JobItem {
  job_id: string;
  original_filename: string;
  file_type: string;
  status: string;
  progress: number;
  stage?: string;
  page_count?: number;
  current_page?: number;
  error?: string;
  mode?: string;
  paddle_device?: string;
  qwen_enabled: boolean;
  created_at: string;
  updated_at: string;
}

// Read API Base URL from environment, fallback to localhost:8000
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export function getApiUrl(path: string): string {
  // If the path is already an absolute URL, return it
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  // Otherwise prefix with API base URL
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${cleanPath}`;
}

export async function submitJob(
  file: File,
  options: {
    mode?: string;
    paddleDevice?: string;
    qwenEnabled?: boolean;
  } = {}
): Promise<JobCreateResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (options.mode) formData.append('mode', options.mode);
  if (options.paddleDevice) formData.append('paddle_device', options.paddleDevice);
  formData.append('qwen_enabled', options.qwenEnabled ? 'true' : 'false');

  const response = await fetch(getApiUrl('/api/jobs'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Failed to submit OCR job');
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(getApiUrl(`/api/jobs/${jobId}`));
  if (!response.ok) {
    throw new Error('Failed to get job status');
  }
  return response.json();
}

export async function getJobResult(jobId: string): Promise<OCRResultResponse> {
  const response = await fetch(getApiUrl(`/api/jobs/${jobId}/result`));
  if (!response.ok) {
    throw new Error('Failed to get job result');
  }
  return response.json();
}

export async function listJobs(limit = 20): Promise<JobItem[]> {
  const response = await fetch(getApiUrl(`/api/jobs?limit=${limit}`));
  if (!response.ok) {
    throw new Error('Failed to list jobs');
  }
  return response.json();
}

export async function deleteJob(jobId: string): Promise<{ job_id: string; deleted: boolean }> {
  const response = await fetch(getApiUrl(`/api/jobs/${jobId}`), {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to delete job');
  }
  return response.json();
}

export function getJobZipUrl(jobId: string): string {
  return getApiUrl(`/api/jobs/${jobId}/download/zip`);
}
