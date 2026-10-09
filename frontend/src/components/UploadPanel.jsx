import { useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/bmp"];

export default function UploadPanel({ onAnalyze, loading, error }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  function selectFile(candidate) {
    if (!candidate) return;
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      return;
    }
    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    selectFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div className="card">
      <h2>Upload MRI Scan</h2>

      {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
      <div
        className={`dropzone${dragOver ? " dragover" : ""}`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload MRI scan image, click or drag and drop"
      >
        <div className="dropzone-icon" aria-hidden="true">
          📤
        </div>
        <div>Click to browse or drag an MRI image here</div>
        <div className="dropzone-hint">PNG, JPG or BMP</div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/bmp"
          onChange={(e) => selectFile(e.target.files?.[0])}
        />
      </div>

      {previewUrl && (
        <>
          <div className="preview-thumb">
            <img src={previewUrl} alt="Selected MRI scan preview" />
          </div>
          <div className="file-meta">{file.name}</div>
        </>
      )}

      <button
        type="button"
        className="btn btn-primary"
        disabled={!file || loading}
        onClick={() => onAnalyze(file)}
      >
        {loading ? "Analyzing…" : "Analyze Scan"}
      </button>

      {error && (
        <p className="error-text" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
