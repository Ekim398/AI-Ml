//api calls
const API_BASE = "https://8000-01ketnz8qfezjczhb9rts2kq8x.cloudspaces.litng.ai"; // host only

export async function inferText(text) {
  const res = await fetch(`${API_BASE}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function inferFrame(fileBlob) {
  const fd = new FormData();
  fd.append("file", fileBlob, "frame.png");

  const res = await fetch(`${API_BASE}/infer-frame`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function analyzeInk(payload) {
  const res = await fetch(`${API_BASE}/analyze-ink`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function ocrFrame(fileBlob) {
  const url = `${API_BASE}/api/ocr`;

  for (let attempt = 0; attempt < 10; attempt++) {
    const fd = new FormData();
    fd.append("image", fileBlob, "frame.png");

    const res = await fetch(url, { method: "POST", body: fd });

    // OCR model still loading on server
    if (res.status === 503) {
      await new Promise((r) => setTimeout(r, 1000));
      continue;
    }

    // Helpful error body when available
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}${text ? `: ${text}` : ""}`);
    }

    return res.json();
  }

  throw new Error("OCR is still warming up (HTTP 503). Try again in a few seconds.");
}

export async function requestHints({ ocr, studentGoal = null, angleUnits = null }) {
  const res = await fetch(`${API_BASE}/api/hints`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ocr,
      student_goal: studentGoal,
      angle_units: angleUnits
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function requestHintsSimple({ fullText, studentGoal = "Solve for x", angleUnits = null }) {
  const payload = { full_text: fullText, angle_units: angleUnits };
  if (studentGoal && typeof studentGoal === "string") payload.student_goal = studentGoal;

  const res = await fetch(`${API_BASE}/api/hints-simple`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const bodyText = await res.text();
  if (!res.ok) {
    console.error("Hints error:", res.status, bodyText);
    throw new Error(`HTTP ${res.status}`);
  }
  return JSON.parse(bodyText);
}
