const API_BASE = "/api";

async function parseJsonOrThrow(res) {
  let body;
  try {
    body = await res.json();
  } catch {
    throw new Error(`Server returned an unexpected response (status ${res.status}).`);
  }
  if (!res.ok) {
    throw new Error(body.error || `Request failed (status ${res.status}).`);
  }
  return body;
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return parseJsonOrThrow(res);
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  return parseJsonOrThrow(res);
}

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("image", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });
  return parseJsonOrThrow(res);
}
