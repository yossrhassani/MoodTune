// Backend connection helper for the Expo app.
// Update BASE_URL to your machine's LAN IP when testing on a physical
// phone (report, Section 7.2: Backend Startup and Network Access) - a
// phone on the same Wi-Fi reaches the PC at e.g. http://192.168.x.x:8000.
// Using an env var keeps this out of source control if it changes often.

import Constants from "expo-constants";

export const BASE_URL =
  Constants?.expoConfig?.extra?.apiBaseUrl || "http://192.168.1.100:8000";

export async function pingBackend() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: "GET" });
    if (!res.ok) return { online: false };
    const data = await res.json();
    return { online: data.status === "ok" };
  } catch (err) {
    return { online: false, error: String(err) };
  }
}

export async function analyzeImage(photoUri) {
  const formData = new FormData();
  formData.append("file", {
    uri: photoUri,
    name: "scan.jpg",
    type: "image/jpeg",
  });

  const res = await fetch(`${BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (!res.ok) {
    throw new Error(`Analyze request failed with status ${res.status}`);
  }
  return res.json();
}
