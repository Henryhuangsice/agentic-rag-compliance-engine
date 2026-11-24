import { MAX_UPLOAD_BYTES, type DocumentInfo } from "@rag/shared";
import { useState, type ChangeEvent } from "react";
import { ApiError, uploadDocument } from "../api/client";

type UploadState =
  | { status: "idle" }
  | { status: "uploading"; name: string }
  | { status: "done"; message: string }
  | { status: "failed"; message: string };

interface Props {
  documents: DocumentInfo[] | null;
  error: ApiError | null;
  onChanged: () => void;
}

export function KnowledgeBase({ documents, error, onChanged }: Props) {
  const [upload, setUpload] = useState<UploadState>({ status: "idle" });

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow picking the same file again
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setUpload({ status: "failed", message: "文件超过 10 MB" });
      return;
    }
    setUpload({ status: "uploading", name: file.name });
    try {
      const r = await uploadDocument(file);
      setUpload({ status: "done", message: `已加入 ${r.source}：${r.chunks} 个片段（总计 ${r.total_chunks}）` });
      onChanged();
    } catch (err) {
      setUpload({ status: "failed", message: err instanceof ApiError ? err.message : "上传失败" });
    }
  };

  const uploading = upload.status === "uploading";
  return (
    <section className="card">
      <div className="kb-head">
        <h2>知识库</h2>
        <label className="btn small upload">
          上传文档
          <input type="file" accept=".md,.txt,.pdf" disabled={uploading} onChange={(e) => void onFile(e)} />
        </label>
      </div>
      <ul className="kb">
        {documents?.map((d) => (
          <li key={d.source}>
            <span className="kb-title" title={d.source}>{d.title}</span>
            <span className="kb-n">{d.chunks} 段</span>
          </li>
        ))}
      </ul>
      {!documents && <p className="hint">{error ? error.message : "加载中…"}</p>}
      <p className={`hint ${upload.status === "failed" ? "upload-error" : ""}`} role="status">
        {upload.status === "idle" && "支持 .md / .txt / .pdf，上传后自动重建索引。"}
        {uploading && `正在上传并重建索引：${upload.name} …`}
        {(upload.status === "done" || upload.status === "failed") && upload.message}
      </p>
    </section>
  );
}
