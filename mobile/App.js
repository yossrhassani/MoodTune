import React, { useRef, useState, useEffect } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { Audio } from "expo-av";
import { pingBackend, analyzeImage } from "./api";

const MOOD_COLORS = {
  calm: "#4fa3ff",
  energetic: "#ff7a3d",
  low: "#6b7bd6",
  positive: "#ffd23d",
  stressed: "#2ee6a6",
};

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState("front");
  const [apiOnline, setApiOnline] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("see");
  const [sound, setSound] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    pingBackend().then((status) => setApiOnline(status.online));
  }, []);

  useEffect(() => {
    return sound ? () => sound.unloadAsync() : undefined;
  }, [sound]);

  if (!permission) return <View style={styles.center}><ActivityIndicator /></View>;

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>MoodTune needs camera access to scan your expression.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  async function scanMood() {
    if (!cameraRef.current) return;
    setLoading(true);
    setResult(null);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
      const data = await analyzeImage(photo.uri);
      setResult(data);
      setActiveTab("see");
    } catch (err) {
      console.warn("Scan failed:", err);
    } finally {
      setLoading(false);
    }
  }

  async function playPreview(url) {
    if (sound) await sound.unloadAsync();
    const { sound: newSound } = await Audio.Sound.createAsync({ uri: url });
    setSound(newSound);
    await newSound.playAsync();
  }

  const accent = result ? MOOD_COLORS[result.mood] || "#2ee6a6" : "#2ee6a6";

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>MoodTune</Text>
        <Text style={styles.status}>{apiOnline ? "API online" : "API offline"}</Text>
      </View>

      {!result && (
        <View style={styles.cameraWrap}>
          <CameraView ref={cameraRef} style={styles.camera} facing={facing} />
          <View style={styles.cameraOverlayCircle} pointerEvents="none" />
        </View>
      )}

      {!result && (
        <View style={styles.controls}>
          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={scanMood}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#062" />
            ) : (
              <Text style={styles.buttonText}>Scan mood</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.secondaryButton}
            onPress={() => setFacing(facing === "front" ? "back" : "front")}
          >
            <Text style={styles.text}>Switch camera</Text>
          </TouchableOpacity>
        </View>
      )}

      {result && (
        <ScrollView style={styles.resultWrap}>
          <View style={[styles.badge, { backgroundColor: accent }]}>
            <Text style={styles.badgeText}>{Math.round(result.confidence * 100)}%</Text>
          </View>
          <Text style={styles.emotionTitle}>
            {result.emotion.charAt(0).toUpperCase() + result.emotion.slice(1)}
          </Text>
          <Text style={styles.moodSubtitle}>Mood group: {result.mood}</Text>

          <View style={styles.tabs}>
            {["see", "meaning", "music", "journey"].map((tab) => (
              <TouchableOpacity
                key={tab}
                style={[styles.tabButton, activeTab === tab && { backgroundColor: accent }]}
                onPress={() => setActiveTab(tab)}
              >
                <Text style={styles.tabText}>{tab}</Text>
              </TouchableOpacity>
            ))}
          </View>

          {activeTab === "see" && (
            <View style={styles.panel}>
              {Object.entries(result.all_scores)
                .sort((a, b) => b[1] - a[1])
                .map(([label, score]) => (
                  <View key={label} style={styles.scoreRow}>
                    <Text style={styles.text}>{label}</Text>
                    <Text style={styles.muted}>{Math.round(score * 100)}%</Text>
                  </View>
                ))}
            </View>
          )}

          {activeTab === "meaning" && (
            <View style={styles.panel}>
              <Text style={styles.headline}>{result.guidance.headline}</Text>
              {result.guidance.steps.map((step, i) => (
                <Text key={i} style={styles.text}>• {step}</Text>
              ))}
              <Text style={styles.affirmation}>{result.guidance.affirmation}</Text>
            </View>
          )}

          {activeTab === "music" && (
            <View style={styles.panel}>
              {result.music.tracks.map((track, i) => (
                <TouchableOpacity
                  key={i}
                  style={styles.track}
                  onPress={() => track.preview_url && playPreview(track.preview_url)}
                >
                  <Text style={styles.text}>{track.title}</Text>
                  <Text style={styles.muted}>{track.artist}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {activeTab === "journey" && (
            <View style={styles.panel}>
              <Text style={styles.muted}>Scan history is stored locally on device.</Text>
            </View>
          )}

          <TouchableOpacity style={styles.button} onPress={() => setResult(null)}>
            <Text style={styles.buttonText}>Scan again</Text>
          </TouchableOpacity>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0f1117", paddingTop: 50, paddingHorizontal: 16 },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#0f1117", padding: 20 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  title: { color: "#fff", fontSize: 20, fontWeight: "700" },
  status: { color: "#9aa0ac", fontSize: 12 },
  text: { color: "#f2f2f2", fontSize: 14, marginVertical: 2 },
  muted: { color: "#9aa0ac", fontSize: 13 },
  cameraWrap: { borderRadius: 16, overflow: "hidden", height: 420 },
  camera: { flex: 1 },
  cameraOverlayCircle: {
    position: "absolute", top: "20%", left: "15%", right: "15%", bottom: "20%",
    borderRadius: 999, borderWidth: 2, borderColor: "rgba(255,255,255,0.5)",
  },
  controls: { marginTop: 16 },
  button: { backgroundColor: "#2ee6a6", padding: 14, borderRadius: 12, alignItems: "center", marginTop: 10 },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: "#062", fontWeight: "700" },
  secondaryButton: { padding: 12, alignItems: "center" },
  resultWrap: { marginTop: 10 },
  badge: { alignSelf: "flex-end", paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  badgeText: { color: "#062", fontWeight: "700", fontSize: 12 },
  emotionTitle: { color: "#fff", fontSize: 28, fontWeight: "700", marginTop: 4 },
  moodSubtitle: { color: "#9aa0ac", marginBottom: 12 },
  tabs: { flexDirection: "row", gap: 6, marginBottom: 10 },
  tabButton: { flex: 1, backgroundColor: "#171a23", paddingVertical: 8, borderRadius: 10, alignItems: "center" },
  tabText: { color: "#f2f2f2", fontSize: 12, textTransform: "capitalize" },
  panel: { backgroundColor: "#171a23", borderRadius: 14, padding: 16, marginBottom: 12 },
  scoreRow: { flexDirection: "row", justifyContent: "space-between", marginVertical: 4 },
  headline: { color: "#fff", fontSize: 16, fontWeight: "600", marginBottom: 8 },
  affirmation: { color: "#9aa0ac", fontStyle: "italic", marginTop: 10 },
  track: { paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: "#232734" },
});
