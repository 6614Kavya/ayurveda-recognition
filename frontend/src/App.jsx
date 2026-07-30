import { useState } from "react"
import axios from "axios"

const API_BASE = "http://localhost:8000"

const styles = {
  container: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #f5f1e6, #e8f5e9)",
    padding: "2rem",
    fontFamily: "'Segoe UI', sans-serif",
    color: "#2e4d34"
  },

  title: {
    textAlign: "center",
    marginBottom: "2rem",
    fontSize: "2.2rem",
    color: "#355e3b"
  },

  card: {
    background: "#ffffffcc",
    backdropFilter: "blur(8px)",
    padding: "1.5rem",
    borderRadius: "16px",
    marginBottom: "1.5rem",
    boxShadow: "0 8px 20px rgba(0,0,0,0.08)"
  },

  sectionTitle: {
    marginBottom: "1rem",
    color: "#4a7c59"
  },

  button: {
    background: "#6b8e23",
    color: "white",
    border: "none",
    padding: "0.6rem 1.2rem",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "0.2s"
  },

  uploadGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem"
  },

  label: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    fontWeight: "500"
  },

  input: {
    padding: "0.4rem",
    borderRadius: "6px",
    border: "1px solid #ccc"
  },

  output: {
    background: "#f0f7f2",
    padding: "1rem",
    borderRadius: "10px",
    marginTop: "1rem",
    fontSize: "0.9rem"
  },

  loading: {
    color: "#6b8e23",
    fontWeight: "bold"
  },

  error: {
    color: "#c0392b",
    fontWeight: "bold"
  }
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function checkHealth() {
    try {
      const res = await axios.get(`${API_BASE}/health`)
      setHealth(res.data)
    } catch (e) {
      setError("Could not reach API — is uvicorn running?")
    }
  }

  async function handleUpload(e, endpoint) {
    const file = e.target.files[0]
    if (!file) return

    const formData = new FormData()
    formData.append("file", file)

    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API_BASE}/predict/${endpoint}`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setPrediction(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || "Upload failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🌿 Ayurveda Plant Recognition</h1>

      <section style={styles.card}>
        <h2 style={styles.sectionTitle}>API Health</h2>
        <button style={styles.button} onClick={checkHealth}>
          Ping Backend
        </button>

        {health && (
          <pre style={styles.output}>
            {JSON.stringify(health, null, 2)}
          </pre>
        )}
      </section>

      <section style={styles.card}>
        <h2 style={styles.sectionTitle}>Test Prediction</h2>

        <div style={styles.uploadGroup}>

          <label style={styles.label}>
            🌸 Flower Image
            <input type="file" accept="image/*"
              style={styles.input}
              onChange={(e) => handleUpload(e, "flower")} />
          </label>

          <label style={styles.label}>
            🍃 Single Leaf Image
            <input type="file" accept="image/*"
              style={styles.input}
              onChange={(e) => handleUpload(e, "single-leaf")} />
          </label>

          <label style={styles.label}>
            🌿 Compound Leaf Image
            <input type="file" accept="image/*"
              style={styles.input}
              onChange={(e) => handleUpload(e, "compound-leaf")} />
          </label>
        </div>
      </section>

      {loading && <p style={styles.loading}>Analyzing plant...</p>}
      {error && <p style={styles.error}>{error}</p>}

      {prediction && (
        <section style={styles.card}>
          <h2 style={styles.sectionTitle}>Prediction Result</h2>
          <pre style={styles.output}>
            {JSON.stringify(prediction, null, 2)}
          </pre>
        </section>
      )}
    </div>
  )
}