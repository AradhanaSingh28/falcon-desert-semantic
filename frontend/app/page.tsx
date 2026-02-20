"use client";

import { useMemo, useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [resultUrl, setResultUrl] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [err, setErr] = useState<string>("");

  const backendUrl = useMemo(() => {
    // Put your backend URL later (HF Space / Render)
    // Example: https://your-space.hf.space/predict
    return process.env.NEXT_PUBLIC_BACKEND_PREDICT_URL || "";
  }, []);

  async function runSegmentation() {
    setErr("");
    setResultUrl("");

    if (!file) {
      setErr("Please upload an image first.");
      return;
    }
    if (!backendUrl) {
      setErr(
        "Backend URL not set. Add NEXT_PUBLIC_BACKEND_PREDICT_URL in .env.local (or Vercel env vars)."
      );
      return;
    }

    try {
      setLoading(true);

      const form = new FormData();
      form.append("file", file);

      const res = await fetch(backendUrl, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Backend error (${res.status}). ${txt}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
    } catch (e: any) {
      setErr(e?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900">
      <div className="max-w-5xl mx-auto p-6">
        <header className="py-6">
          <h1 className="text-3xl font-bold">Offroad Semantic Segmentation</h1>
          <p className="text-zinc-600 mt-2">
            Upload an offroad image → get a predicted segmentation mask.
          </p>
        </header>

        <section className="bg-white rounded-2xl border p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Upload image</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  setFile(f);
                  setResultUrl("");
                  setErr("");
                  if (f) setPreviewUrl(URL.createObjectURL(f));
                  else setPreviewUrl("");
                }}
              />
              <p className="text-xs text-zinc-500">
                Tip: Use any image from your test set (Color_Images).
              </p>
            </div>

            <button
              onClick={runSegmentation}
              disabled={!file || loading}
              className="px-5 py-3 rounded-xl bg-black text-white disabled:opacity-50"
            >
              {loading ? "Running..." : "Run Segmentation"}
            </button>
          </div>

          {err && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              {err}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div>
              <h2 className="font-semibold mb-2">Input</h2>
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="Input preview"
                  className="rounded-xl border w-full"
                />
              ) : (
                <div className="rounded-xl border bg-zinc-50 p-6 text-zinc-500">
                  Upload an image to preview it here.
                </div>
              )}
            </div>

            <div>
              <h2 className="font-semibold mb-2">Predicted Mask</h2>
              {resultUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={resultUrl}
                  alt="Predicted mask"
                  className="rounded-xl border w-full"
                />
              ) : (
                <div className="rounded-xl border bg-zinc-50 p-6 text-zinc-500">
                  Run segmentation to see the predicted mask here.
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-2xl border p-4">
            <div className="text-sm text-zinc-500">Model</div>
            <div className="font-semibold">U-Net (Weighted CE + Dice)</div>
          </div>
          <div className="bg-white rounded-2xl border p-4">
            <div className="text-sm text-zinc-500">Input</div>
            <div className="font-semibold">RGB Image</div>
          </div>
          <div className="bg-white rounded-2xl border p-4">
            <div className="text-sm text-zinc-500">Output</div>
            <div className="font-semibold">Segmentation Mask (Class IDs)</div>
          </div>
        </section>

        <footer className="py-8 text-xs text-zinc-500">
          Set backend URL via{" "}
          <span className="font-mono">NEXT_PUBLIC_BACKEND_PREDICT_URL</span>.
        </footer>
      </div>
    </main>
  );
}
